# 7. 前向计算（Prefill / Decode）

## 7.1 `Forward` 主循环（`Qwen35Model.cs`）

```
Forward(tokens):
  _scratchUsed = 0
  seq = tokens.Length
  start = _cacheLen
  EnsureCache(start + seq)      // 需要时倍增扩容 KV
  EnsureScratch(seq, start+seq) // 按实际尺寸整备 arena
  h = Bump(seq·hidden)          // 激活
  Embed(tokens, h)
  layerMark = _scratchUsed
  if seq == 1:
      for each layer: _scratchUsed=layerMark; DecodeBlock(h, layer, start)
  else:
      for each layer: _scratchUsed=layerMark; PrefillBlock(h, layer, seq, start)
  normed = Bump(seq·hidden); Rms("output_norm.weight", h, normed, ...)
  logits[vocab] = Linear(last_row, "output.weight", fallback="token_embd.weight")
  _cacheLen += seq; _cacheTokens.AddRange(tokens)
  return logits
```

注意：**prefill 一次处理整段 suffix**，decode 每次只喂 1 个 token。每层内部从
`layerMark` 重新 `Bump`，保证层间复用同一块 scratch 而不越界。

`Q35_DIAG=1` 时每 3 层打印最后 token 隐藏态 RMS 范数（归一化到 /hidden 的
`|h|`），最后打印整体 `|h|`，用于定位数值发散。

## 7.2 单层统一结构（`PrefillBlock` / `DecodeBlock`）

```
n1 = Rms(attn_norm, h)
token mixer:
   线性层 → PrefillLinearAttn / DecodeLinearAttn
   全注意力 → PrefillAttention / DecodeAttention
h += attn                          // 残差 1
n2 = Rms(post_attention_norm, h)
FFN:
   MoE → PrefillSparseMoe / DecodeSparseMoe
   dense → gate=W_g·h, up=W_u·h, silu(gate)·up, down=W_d·(silu(gate)·up)
h += ffn_out                        // 残差 2
```

dense FFN 的 `gate/up` 两个 GEMV 共享**同一份** Q8 激活（`DecodeGateUp` 一次
`Q8K.QuantizeRow` 后两次 `GemvPrequant`；prefill 用 `MulMatPanel`/`MulMatQ2`/
`MulMatSTQ` 的 `QuantizeAndGemm`，一个并行区段量化 + 两个/三个 panel GEMM）。
`ffn_down` 用 `SiluQuantizeAndGemm` 融合 SiLU·up 进量化。

## 7.3 全注意力层

### 7.3.1 投影与 q|gate 解交错

`attn_q.weight` 输出宽 `2·qDim`，头内交错 `[q_h0, gate_h0, q_h1, gate_h1, …]`。
GEMM/GEMV 之后按头解交错到两个独立的 `q` / `gate` 缓冲（head-major
`[seq][heads][dim]`）。`attn_k` / `attn_v` 直接写进 KV 缓存 `start..start+seq`。

### 7.3.2 RoPE 与 per-head RMSNorm

`Ops.NeoXRoPE`：对 q、k 按 `RopeDim`（部分旋转）应用，cos/sin 表按 `(pos, freq)`
缓存（`EnsureRopeTable`，freq_i = base^(-i/half)，容量 ≥4096 自动增长）。随后
`attn_q_norm.weight` / `attn_k_norm.weight` 按**头**做 zero-centered RMSNorm
（`Rms("...", q, q, seq·heads, dim)`）。

### 7.3.3 注意力评分 / softmax / 加权

```
scale = 1/√dim
AttentionScores: 对每个头 h（kv 头 = h/group, group=heads/kvHeads），
  逐 q 行对可见 k 前缀做点积；AVX2+FMA 时用 Dot2x4（2 q 行 × 4 k 行）批处理
SoftmaxCausal: 只归一 [0, startPos+qt+1) 前缀，尾部精确写 0
AttentionCombine: outH[qt,h] = Σ_kt scores·v[kt,kvh]；Axpy64 让 64 维输出驻留寄存器
```

GQA：`group = heads/kvHeads`，每个 kv 头服务一组 q 头。

### 7.3.4 sigmoid 门控与输出

```
ao *= 1/(1+exp(-gate))     // 逐元素（prefill 用批量，decode 单行）
output = ao · attn_output.weightᵀ
```

sigmoid 门控 `out·sigmoid(gate)` 是 Qwen3.5 相对 Qwen2/3 全注意力层的核心改动。

## 7.4 Gated-DeltaNet 线性注意力层

### 7.4.1 投影

| 张量 | 维度 | 语义 |
|------|------|------|
| `attn_qkv.weight` | → `convDim` | 拼接的 q/k/v（`LinearConvDim = 2·keyDim + valueDim`） |
| `attn_gate.weight` | → `valueDim` | z（输出门控） |
| `ssm_beta.weight` | → `nHeads` | β |
| `ssm_alpha.weight` | → `nHeads` | α |
| `ssm_a`（F32） | `nHeads` | log A → **−exp(A)**（GGUF 已烘烤） |
| `ssm_dt.bias`（F32） | `nHeads` | dt 偏置 |
| `ssm_conv1d.weight`（F32） | `[kernelSize][convDim]` | 因果深度卷积核 |
| `ssm_norm.weight`（F32） | `vHeadDim` | 门控 RMSNorm 权重 |
| `ssm_out.weight` | → hidden | 输出投影 |

### 7.4.2 因果卷积

* **prefill**：qkv 先转置成 channel-major `[convDim][seq]`，`CausalConv1d`
  逐通道做 `out[t] = silu(Σ_j kernel[j][c]·in[t-j])`（左补 kernelSize-1）。随后把
  最后 `stateLen = kernelSize-1` 个**原始输入**值写回 `_linConv[l]`。
* **decode**：`CausalConv1dUpdate` 用滚动窗口 `convState[c][0..stateLen)`（index 0
  最旧）一次算出 `silu(Σ_j kernel[j][c]·state[c][j] + kernel[stateLen][c]·in[c])`，
  并把新输入推进窗口。

### 7.4.3 拆分与头重复

conv 输出 `[keyDim | keyDim | valueDim]`。按参考实现把 q/k 头**重复**到 value 头
数：`nKHeads = group_count`，`nVHeads = time_step_rank`，
`repeat = nVHeads / nKHeads`，`vh = kh·repeat + rep`。q/k 展开成
`[seq][nHeads][headDim]`，v 是 `[seq][valueDim]`。

### 7.4.4 β 与衰减

```
betaP = sigmoid(bOut)
gP[head] = ssm_a[head] · softplus(aOut + ssm_dt.bias[head])   // ≤ 0
decay = exp(gP)
```

`ssm_a` 已存 −exp(A)，所以**不再做 exp**；`g = −exp(A)·softplus(α+dt)` 恒 ≤ 0。

### 7.4.5 Delta-rule 递推（`GatedDeltaNetOps.RecurrentGatedDeltaRule`）

对每个头 `h`（状态 `S[h] : [headDim][vHeadDim]`）逐 token `i`：

```
l2normQk（Qwen3.5 恒 true）：
  q̂ = q·(scaleQ/‖q‖),  k̂ = k/‖k‖,  scaleQ = 1/√headDim
decay = exp(g[i][h])
DeltaStep(S, q̂, k̂, v, decay, β):
  1) S *= decay
  2) kvMem[b] = Σ_a S[a,b]·k̂[a]          // read
  3) delta[b] = (v[b] − kvMem[b])·β
     S[a,b] += k̂[a]·delta[b]             // write
  4) out[b] = Σ_a S[a,b]·q̂[a]            // read
```

单测 `GatedDeltaNetOpsTests` 用独立标量参考实现对拍（含 L2 归一化分支）。

### 7.4.6 门控归一化与输出

```
gated = GatedRmsNorm(out, ssm_norm, z, eps)   // norm(out)·silu(z)
output = gated · ssm_out.weightᵀ
```

## 7.5 KV 缓存与前缀复用

* 全注意力层 KV 缓存：`[pos][kvStride]` f32，逐层分配，`start` 作为本轮写入起点。
* `KvCacheAlign.Plan(cached, prompt)`：
  * 找公共前缀长度 `common`；
  * 若 `common == prompt.Length`（全命中），截断到 `len-1`、重放最后 token 刷新
    logits；
  * 否则截断到 `common`，suffix 从 `prompt[common..]` 开始。
  * 返回 `PromptReuse(TruncateTo, SuffixStart)`。
* `Qwen35Model.AlignPrompt` 据此 `TruncateCache(TruncateTo)` 并给出 suffix。
  这使得多轮对话复用已缓存前缀，仅重算新增部分；同时更新线性层的递推状态——
  **注意**：截断缓存只作用于全注意力 KV；线性层状态 `S` 与卷积窗口在截断时不回滚，
  因此当前实现要求对话进程内缓存只增（`TruncateCache` 不允许超过 `_cacheLen`），
  `/reset` 通过 `ResetCache` 全清。

## 7.6 数值/性能要点

* softmax 用 `FastExp.ExpAvx2`，被减量 floor 到 −80 避免下溢污染归一。
* 所有头/行循环按 worker 等分切片，`CpuThreadPool.For` 派发。
* attention scores 的行内 `Dot2x4`/`DotF32x4` 批处理减少 K 重载；combine 的
  `Axpy64` 把 64 个输出维保持在寄存器。