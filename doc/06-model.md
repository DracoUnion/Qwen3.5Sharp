# 6. 模型层基础（`Qwen35Model` 的配置 / 权重 / 分派）

## 6.1 `ModelConfig`（`ModelConfig.cs`）

从 GGUF 元数据抽取全部超参。架构读取 `general.architecture`，支持 `qwen35` 与
`qwen35moe`。

### 主要字段

| 组 | 字段 | GGUF 键 |
|----|------|---------|
| 通用 | `Architecture` / `NumLayers` / `HiddenSize` / `FfnSize` / `VocabSize` / `ContextLength` / `Eps` | `embedding_length` / `block_count` / `feed_forward_length` / `vocab_size` / `context_length` / `layer_norm_rms_epsilon` |
| 全注意力 | `NumHeads` / `NumKvHeads` / `HeadDim` / `RopeDim` / `RopeBase` | `attention.head_count` / `head_count_kv` / `key_length`（缺省 = hidden/heads）/ `rope.dimension_count`（缺省 = headDim）/ `rope.freq_base` |
| 线性注意力 | `SsmConvKernel` / `SsmInnerSize` / `SsmStateSize` / `SsmTimeStepRank` / `SsmGroupCount` | `ssm.conv_kernel`（缺省 4）/ `ssm.inner_size` / `ssm.state_size`（缺省 headDim）/ `ssm.time_step_rank` / `ssm.group_count` |
| MoE | `IsMoe` / `NumExperts` / `NumExpertsPerTok` / `MoeFfnSize` / `MoeSharedFfnSize` / `MoeSharedCount` / `ExpertWeightsScale` | `expert_count` / `expert_used_count` / `expert_feed_forward_length` / `expert_shared_feed_forward_length` / `expert_shared_count` / `expert_weights_scale` |
| 层类型 | `IsRecurrent[]`（true=线性注意力） | 见下 |

### 派生维度

```
LinearConvDim  = 2·SsmGroupCount·SsmStateSize + SsmInnerSize   // qkv 拼接宽
LinearKeyDim   =   SsmGroupCount·SsmStateSize                  // k 宽
LinearValueDim =   SsmTimeStepRank·SsmStateSize                // v / z / out 宽
LinearQkvDim   = 2·SsmStateSize·SsmGroupCount + SsmInnerSize
FullQDim       = NumHeads·HeadDim
FullKvDim      = NumKvHeads·HeadDim
```

### 层类型判定（关键）

不再依赖 `full_attention_interval` 公式硬算，而是**看张量**：

* `blk.L.attn_qkv.weight` 存在 → 线性注意力层（Gated-DeltaNet）。
* `blk.L.attn_q.weight` 存在 → 全注意力层。
* 都没有 → 回退公式 `(L+1) % full_attention_interval != 0`。
* 若整个文件都没有层张量（仅 header），全部用公式。

这样能容纳不完全遵循间隔模式的新 checkpoint。MoE 判定 `arch == "qwen35moe"`，
`expert_used_count` 缺省按 MoE 给 8。

## 6.2 权重加载（`Qwen35Model.LoadWeights`）

### `Weight` 结构体（union 式）

```csharp
struct Weight {
    GgmlTensorType Type;
    BlockQ4K*   Q4;      BlockQ4Kx8*  Packed;   BlockQ4Kx8Meta* Meta;
    BlockQ5K*   Q5;
    BlockQ6K*   Q6;      BlockQ6Kx8*  Packed6;
    BlockQ2_0C* Q2;      BlockQ2x8*   Packed2;
    BlockSTQ1_0* STQ;    BlockSTQ1_0x8* PackedSTQ;
    BlockIQ4XS* Q4xs;
    float*      F32;
    int NIn, NOut;
    int NExpert;  long ExpertStride;   // 3D MoE 专家张量
    int Count;                          // F32 元素数
}
```

只使用与 `Type` 匹配的字段；dispatch 靠 `switch (w.Type)`。

### 加载流程

遍历 `_gguf.Tensors`：

* **F32**：拷贝进 `NativeBuffer`（`Count = NumElements`）。MoE router、shared gate
  等小张量、以及全部 `*_norm.weight` / `ssm_a` / `ssm_dt.bias` 都是 F32。
* **量化张量**：
  1. 由 `info.Shape[0]`（nIn）判定块数 `nb` 与单块字节 `blockBytes`，整段拷进
     `NativeBuffer`。
  2. 若 `nOut >= 8 && nExpert == 1`，额外构建 panel：
     * Q4_K → `RepackQ4K.Rows`（`Q4Kx8`）+ `RepackQ4K.BuildMeta`（`Q4Kx8Meta`）。
     * Q6_K → `RepackQ6K.Rows`（`Q6Kx8`）。
     * Q2_0C → `RepackQ2.Rows`（`Q2x8`）。
     * STQ1_0 → `RepackSTQ.Rows`（`STQ1_0x8`）。
  3. 3D 专家张量（`Shape.Length == 3`）记录 `NExpert` 与
     `ExpertStride = nOut·nb·blockBytes`（专家之间的字节步长），不重打包——MoE
     逐专家做 GEMV。

`token_embd.weight` 只要求 `nIn`（hidden）是 256 的倍数；它按行反量化进隐藏态。

### 生命周期与所有权

* `_buffers` 记录所有 `Rent` 出的 `NativeBuffer`，`Dispose` 统一释放。
* `_gemmScratch` 是内核级 `ScratchArena`，供 GEMV 的 Q8 行、GEMM 的 Q8 面板与
  补零暂存。
* KV 缓存与线性层状态在构造函数分配（见下节）。

## 6.3 缓存与状态分配

### 全注意力 KV 缓存

* 只给 `IsRecurrent == false` 的层分配：`_cacheK[l]` / `_cacheV[l]`，
  容量 `_cacheCap = min(ContextLength, 4096)`，行距 `kvStride = FullKvDim`。
* `EnsureCache`：按需**倍增扩容**（`next = max(needed, cap*2)`，封顶
  `ContextLength`），新 buffer 拷入旧内容后原子替换并释放旧 buffer。中途异常回滚。
* 布局：`[pos][kvStride]`，pos 从 0 开始；`start` 索引传遍前向。

### 线性注意力状态

* `_linState[l]`：`[nHeads=SsmTimeStepRank][headDim=SsmStateSize][vHeadDim=SsmStateSize]`
  f32，delta-rule 状态矩阵 `S`。
* `_linConv[l]`：`[stateLen = max(1, convKernel-1)][convDim = LinearConvDim]`
  f32，滚动卷积窗口（index 0 = 最旧）。
* 全部从两个大 `NativeBuffer` 按层切片分配；`ResetCache` 把它们清零。

### 模型级 scratch

`EnsureScratch(seq, kvLen)` 一次性估算单层最坏占用（线性层内部、全注意力层内部、
MoE 内部、FFN、norm），取三者中**线性/全注意力取 max、再与 MoE/FFN/norm 相加**，
加上 4 MB 余量。`Bump(bytes)` 按 64B 对齐从 `_scratch` 顺序切，`_scratchUsed`
在每层开头回卷到 `layerMark`。arena 不够时按估算重建；`Bump` 越界抛
`InsufficientMemoryException` 提示重算估算。

## 6.4 线性分派（`Qwen35Model.Decode.cs`）

| 方法 | 语义 | 分派目标 |
|------|------|----------|
| `Linear(input, name, out, nIn, nOut, tokens, fallback)` | prefill/通用矩阵乘 | `MulMatQ4K.Gemm` / `MulMatQ2.Gemm` / `MulMatSTQ.Gemm` / `Q5K.Gemm` / `Q6K.Gemm` / `IQ4XS.Gemm` / `F32Gemm` |
| `DecodeLinear(...)` | decode 单行 GEMV | 对应 `*Gemv` |
| `GemvPrequant(w, x, y, …)` | 使用**已有** Q8 行做 GEMV（gate/up/q/k/v 复用） | `MulMat*.GemvPrequant` / `Q5K`/`Q6K`/`IQ4XS.GemvPrequant`；默认 Q4_K |
| `F32Gemm(w, x, y, …)` | 纯 f32 `y = x·Wᵀ`，行主序 `W[i+o·nIn]` | 逐 token 逐行 `Ops.DotF32` |
| `Rms` / `RmsSerial` | zero-centered RMSNorm | `Qwen35Ops.RmsNormZeroCentered` |

`Linear` 支持 `fallback`（如 LM head 用 `token_embd.weight` 共享）。`F32Gemm`
用于 router 等小张量，避免为它们引入内核。

`Embed` 按 `token_embd.weight` 类型分派反量化（Q4_K/Q5_K/Q6_K/Q2_0C/STQ1_0/
IQ4_XS/F32），`tokens>8` 时并行。

## 6.5 公共 API（供 CLI / 测试）

* `Forward(int[] tokens)` → `float[] logits`（返回**最后** token 的 logits，但内部
  计算整个序列并更新缓存）。
* `AlignPrompt(ReadOnlySpan<int> promptIds)` → `PromptAlignment(Cached, Suffix)`：
  前缀复用规划。
* `ResetCache` / `TruncateCache(len)`。
* 调试辅助：`EmbedRow(token, dst)`、`DequantRow(name, row, dst)`、
  `LmHeadDot(h, row)`。
* `ThreadCount` / `ThreadAutoHint`、静态 `ProfileEnabled` + `Ticks*` 计数器与
  `ResetProfile()`。

## 6.6 逐点算子（`Qwen35Ops.cs` / `Ops.cs`）

* **zero-centered RMSNorm**（Qwen3/Qwen3.5）：`y = x·rsqrt(mean(x²)+eps)·(1+w)`，
  权重初始化为 0 故用 `1+w`；与 HyMT2Sharp 的 Llama 风格 `x·rsqrt·w` 区分。
  用于 decoder 输入/输出 norm、最终 norm、per-head Q/K norm。
* **Gated RMSNorm**：`(x·rsqrt(mean(x²)+eps)·w) · silu(gate)`，用于 Gated-DeltaNet
  输出。
* **Sigmoid / Softplus / Log1pSafe**：`softplus(x)=log(1+e^x)` 数值稳定分三段
  （x>20 直接 x，x<-10 用 e^x，中间用 `Log1pSafe` 泰勒级数）。
* **Ops**（通用）：Llama 风格 `RmsNorm`、`NeoXRoPE`（cos/sin 缓存表）、
  `SoftmaxCausal`（只归一可见前缀、尾部精确清零）、`AttentionScores`/`AttentionCombine`、
  `DotF32`/`AxpyF32`/`SiLUMul`/`AddInPlace` 及各自 AVX2 变体。