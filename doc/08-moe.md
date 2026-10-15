# 8. MoE 稀疏 FFN（`Qwen35Model.Moe.cs`，arch `qwen35moe`）

## 8.1 数学

```
router_logits = x @ ffn_gate_inpᵀ            [tokens, num_experts]
probs = softmax(router_logits)
top-k 索引与权重 → 重归一化权重和为 1
out = Σ_k expert_weight_k · Expert_k(silu(gate_k(x)) · up_k(x))
shared = sigmoid(x @ ffn_gate_inp_shexpᵀ) · SharedFFN(x)   // 若存在共享专家
out += shared
```

## 8.2 权重布局

### 3D 专家张量（`[nIn, nOut, num_experts]`）

| 张量 | 形状 | 语义 |
|------|------|------|
| `blk.L.ffn_gate_exps.weight` | `[hidden, moeFfn, experts]` | 每个专家的 gate |
| `blk.L.ffn_up_exps.weight` | `[hidden, moeFfn, experts]` | 每个专家的 up |
| `blk.L.ffn_down_exps.weight` | `[moeFfn, hidden, experts]` | 每个专家的 down |

加载期只记录 `NExpert` 与 `ExpertStride = nOut·nb·blockBytes`（专家间字节步长），
**不重打包**成 panel——MoE 在 decode 阶段只做逐专家 GEMV，且每次只激活 top-k 个
专家。`ExpertBase(w, exp) = base + exp·ExpertStride` 定位专家切片。

### 共享专家与 router（2D / F32）

| 张量 | 形状 | 语义 |
|------|------|------|
| `blk.L.ffn_gate_inp.weight` | `[hidden, num_experts]` F32 | router |
| `blk.L.ffn_gate_inp_shexp.weight` | `[hidden, 1]` F32 | 共享门控 logit |
| `blk.L.ffn_gate_shexp.weight` | `[hidden, sharedFfn]` | 共享 gate |
| `blk.L.ffn_up_shexp.weight` | `[hidden, sharedFfn]` | 共享 up |
| `blk.L.ffn_down_shexp.weight` | `[sharedFfn, hidden]` | 共享 down |

## 8.3 Prefill（`PrefillSparseMoe`）

1. **router**：`Linear(ffn_gate_inp, x) → routerLogits[rows, experts]`。
2. **softmax + top-k**：`TopKSoftmax` 先整行 softmax（减 max、`exp`、归一），再
   贪心选 top-k（nUsed ≤ 8，简单线性扫描；选中项置 −1 排除），最后把选中的
   weights 重归一化为和 1。
3. **逐 token 逐专家**：对 `row r`、`k`：
   * `gate`/`up` = `ExpertGateUp(x, layer, exp, …)`（两个 GEMV，共用一次 Q8
     量化——`IQ4XS.Gemv`/`Q5K.Gemv`/`MulMatQ4K.Gemv` 各自内部量化一次；
     当前实现为每次 GEMV 独立量化，未来可合并）。
   * `Ops.SiLUMul(gate, up)`；`ExpertDown(gate, …)` → `down`。
   * `output[r] += w · down`。
4. **共享专家**：若 `ffn_gate_shexp.weight` 存在：
   * `sharedGateLogit = F32Gemm(ffn_gate_inp_shexp, x)`（每行一个标量）。
   * 共享 FFN：`Linear(ffn_gate_shexp/up_shexp) → SiLUMul → Linear(ffn_down_shexp)`。
   * `output[r] += sigmoid(sharedGateLogit[r]) · sharedOut[r]`。

## 8.4 Decode（`DecodeSparseMoe`）

与 Prefill 完全同构，只是 `rows=1`：router → top-k → 逐专家 GEMV → 共享专家。
decode 路径全部走 `DecodeLinear`（单行 GEMV）。

## 8.5 性能特征与取舍

* **每次只算 top-k 个专家**（典型 8 / 256），避免 256 个专家全矩阵乘。
* 3D 张量逐专家 GEMV 无法复用 Q8 面板；这是 MoE 当前的主要热点，优化方向是
  **按专家批处理**（同时处理多个 token 选中同一专家的块），属于未来工作。
* router/shared 门控用 F32（`F32Gemm`），避免为 tiny 张量建内核。
* `ExpertWeightsScale` 已从 GGUF 读入但当前**未应用**到专家输出（`qwen35moe`
  若需要乘法缩放需在 `output[r] += w·down` 处乘 `expert_weights_scale`）。

## 8.6 正确性要点

* top-k 权重重归一化必须在 softmax **之后**再做，保证 Σ w = 1。
* 共享专家的 sigmoid 门控与主 MoE 输出相加顺序与参考实现一致。
* 专家切片字节偏移用 `ExpertStride` 计算而非按 `NOut` 线性——因为 3D 张量
  `[nIn, nOut, experts]` 的专家沿第三维，切片间距是 `nOut·nb·blockBytes`。