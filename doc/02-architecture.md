# 2. 总体架构

## 2.1 分层与依赖

```
┌──────────────────────────────────────────────────────────┐
│  Qwen35Sharp.Cli  Program.cs                               │
│  参数解析 · REPL · 流式输出 · 计时统计                      │
└───────────────▲──────────────────────────────────────────┘
                │ 调用模型 API
┌───────────────┴──────────────────────────────────────────┐
│  Qwen35Sharp.Model                                        │
│  Qwen35Model（prefill/decode/缓存） · ModelConfig          │
│  Qwen35Ops / Ops / GatedDeltaNetOps（逐点 + 递推）          │
│  BpeTokenizer · ChatTemplate · KvCacheAlign               │
└───────▲───────────────────────────▲──────────────────────┘
        │ 引用                       │ 引用
┌───────┴───────────┐      ┌─────────┴──────────────────┐
│ Qwen35Sharp.Kernels│      │ Qwen35Sharp.Gguf           │
│ GEMM/GEMV 内核     │      │ 内存映射 GGUF v2/v3 读取器  │
│ 线程池 / 拓扑 / 内存│      │ 元数据 + 张量寻址           │
└────────────────────┘      └───────────────────────────┘
```

依赖方向严格单向：**Cli → Model → {Kernels, Gguf}**。Model 依赖 Kernels 的计算内核
与 Gguf 的文件读取；任何一层都不反向依赖上层。

## 2.2 各项目职责

| 项目 | 职责 | 关键类型 |
|------|------|----------|
| `Gguf` | 文件格式层：把 GGUF 的字节变成可寻址的张量 | `GgufFile`, `GgufTensorInfo`, `GgmlTensorType` |
| `Kernels` | 计算层：量化点积/GEMV/GEMM、逐点 SIMD 算子、并发与内存原语 | `Q4K`/`Q5K`/`Q6K`/`Q8K`/`Q2_0C`/`STQ1_0`/`IQ4XS`, `MulMat*`, `GemmQ4K`/`GemmQ6K`, `Q2Panel`/`STQPanel`, `Ops` 不在此（在 Model）… |
| `Kernels`（续） | 基础设施 | `CpuThreadPool`, `CpuTopology`, `NativeBuffer`, `ScratchArena`, `FastExp`, `BlockLayout` |
| `Model` | 模型层：配置抽取、权重加载、分层前向、缓存状态、分词、模板 | `Qwen35Model`（partial：Decode/Prefill/Moe）, `ModelConfig`, `Ops`, `Qwen35Ops`, `GatedDeltaNetOps`, `BpeTokenizer`, `ChatTemplate`, `KvCacheAlign` |
| `Cli` | 运行时壳：生成循环、输出 | `Program`（top-level statements） |
| `Tests` | xunit 单元测试与对拍测试 | `FastExpTests`, `GatedDeltaNetOpsTests`, `GgufTypeTests`, `IQ4XSTests`, `KvCacheAlignTests` |

## 2.3 一次推理（自回归一步）的数据流

```
用户文本
  │ BpeTokenizer.Encode
  ▼
token ids[]
  │ KvCacheAlign.Plan（前缀复用，计算需要重放的 suffix）
  ▼
Qwen35Model.Forward(suffix)
  │  Embed → [tokens, hidden]          （Q4_K/Q5_K/Q6_K/Q2_0C/STQ1_0/IQ4_XS/F32 反量化）
  │  逐层：
  │    attn_norm（RMSNorm）
  │    ├─ 全注意力：q|gate / k / v → RoPE → 头RMSNorm → scores → softmax → combine → gate
  │    └─ 线性注意力：qkv/z/beta/alpha → conv1d(SiLU) → 头重复 → delta-rule 递推 → gated RMSNorm
  │    → 残差
  │    post_attention_norm → FFN（dense SiLU 或 MoE）→ 残差
  │  output_norm → [hidden]
  ▼
logits[vocab]
  │ ArgMax（贪心）
  ▼
下一个 token id
```

关键点：**`Forward` 一次处理一段 token 序列**。`seq==1` 时走逐层单 token
`DecodeBlock`；`seq>1` 时走批量 `PrefillBlock`。自回归的第 N 步 `Forward(tokens=[t])`
就是 decode。

## 2.4 关键设计决策

### D1. 加载时一次性重打包权重成 panel 格式

GGUF 中的权重是**行主序**的 K 块（每个 super-block 256 个值，带自己的 d/dmin/
scales）。直接用它做 GEMM 需要在最内层循环解码位字段。项目在 `LoadWeights` 阶段就把
每 8 个输出行（列）重打包成 panel（`BlockQ4Kx8` / `Q6Kx8` / `Q2x8` / `STQ1_0x8`），
并给 Q4_K 额外生成**预解码**的 meta（f32 d/dmin + int16 scales/mins）。

* 好处：热循环（GEMM/GEMV）完全不做位手术，只做乘加。
* 代价：加载时内存翻几倍（面板格式比源格式大，例如 Q4_K 128→1152×nOut/8），但
  属于一次性、只读、可上线接受。原格式仍保留用于 decode 的 GEMV 尾部与小矩阵。
* Q4_K 的 meta 也解决列序问题：`vphaddw` 得到的列序是 `[0 1 4 5 2 3 6 7]`，
  在存储时用一次 `PermuteVar8x32` 修正。

### D2. decode = GEMV，prefill = panel GEMM，激活统一 Q8_K 量化

* **decode（1 token）**：把激活向量 `Q8K.QuantizeRow` 成 `BlockQ8K`，然后对每行
  权重做点积（AVX2 版 `VecDotQ4K.Dot` / `Q6K.Dot` / `Q2_0C.Dot` / `IQ4XS.Dot` /
  `STQ1_0.Dot`）。一次量化，`gate`/`up`/`q`/`k`/`v` 等多个 GEMV 共用同一个 Q8 行。
* **prefill（N tokens）**：一次并行量化 4 个 token 为一组
  （`QuantizeQ8Kx4.Quantize4x8` → `BlockQ8Kx4`），用 RAM 中常驻的 panel 权重做
  4×8 GEMM。`MulMatPanel` / `MulMatQ2` / `MulMatSTQ` 支持**一个并行区段共享一份
  Q8 面板、连续跑至多三个矩阵**（如 `ffn_gate`/`ffn_up`，或 `attn_q`/`attn_k`/
  `attn_v`），并在量化与 GEMM 之间用自旋 barrier 同步，完全对齐 llama.cpp 的
  `mul_mat` 行为。
* ffn_down 有**融合版本** `SiluQuantizeAndGemm`：SiLU(gate)·up 直接在量化循环里
  计算，避免先写回 f32 再量化。

### D3. 顺序递推 Gated-DeltaNet

线性注意力层维护每层的 `[nHeads][headDim][vHeadDim]` 状态矩阵 `S`，prefill 在序列
上逐 token 执行 delta 规则（见 `07-forward.md`），decode 单步执行。不做分块
（chunk）——两者数学等价，分块仅为长序列 prefill 的内存优化，未来工作。

### D4. 内存策略：mmap + NativeBuffer 拷贝 + scratch arena

* GGUF 用 `MemoryMappedFile` 只读映射，权重按需**整块拷贝**进对齐的
  `NativeBuffer`（64B 对齐）。kv 元数据扫一遍即丢弃。
* 激活/中间张量统一从 `_gemmScratch`（模型级 `ScratchArena`）或 `Qwen35Model` 的
  `_scratch` 大块 arena 中按 64B 对齐 `Bump` 出。arena 在每次 `Forward` 开头按
  实际 `seq` 和 KV 长度估算所需大小并懒增长，一次前向内不再增长，避免前向中途
  指针失效。层间复用（`_scratchUsed = layerMark` 回卷）。
* KV 缓存仅分配给**全注意力层**；线性注意力层持 `[head][kDim][vDim]` 状态与滚动
  卷积窗口，大小固定。
* 主要权衡：模型级 `_scratch` 是单调增长不复用的（用更大就换新 buffer，旧 buffer
  进入 `_buffers` 由 `Dispose` 统一释放），但与“每层重复 Bump、层间回卷”配合，
  单层最大并发缓冲远小于整个上下文窗口。

### D5. 线程模型：专用工作线程 + job-id 轮询 + 自旋 barrier

`CpuThreadPool` 在构造时按物理 P-core 数生成固定线程，用 `jobId + AutoResetEvent`
实现极低开销的任务派发，`Barrier()` 实现量化和 GEMM 之间的同步。默认线程数来自
`CpuTopology` 的物理 P-core 计数，而不是 `Environment.ProcessorCount`（防止
SMT/E-core 拖累 lockstep GEMM）。详见 `05-kernels.md`。

### D6. 前缀缓存复用（Prompt caching）

`KvCacheAlign.Plan` 找出现有 `_cacheTokens` 与提示词的公共前缀，把公共前缀从本次
`Forward` 里删掉（`TruncateCache` 到公共前缀长度），只对 suffix 重新前向，并更新
缓存状态到新长度。全命中时重放最后一个 token 以刷新 logits。这让多轮对话无需每次
重算已缓存的前缀。

### D7. 诊断与剖析内建

* 静态时钟计数器（`TicksQ4/Q2/STQ/Q6/AttnScore/…`）在 `ProfileEnabled` 时逐个算子
  累加 `Stopwatch.GetTimestamp()` 差，方便做算子级剖面。
* `Q35_TRACE=1` 打印每层每个子步骤进入/完成的跟踪行。
* `Q35_DIAG=1` 每 3 层打印最后一 token 隐藏态 RMS 范数，定位数值发散层。

## 2.5 构建产物命名

所有程序集以 `Flygon.` 前缀命名（如 `Flygon.Qwen35Sharp.Model.dll`），
`RootNamespace` 同步为 `Flygon.<Project>`，因此源码内命名空间为 `Flygon.Gguf`、
`Flygon.Kernels`、`Flygon.Model`。