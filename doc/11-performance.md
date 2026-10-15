# 11. 性能设计与剖析

## 11.1 性能设计要点回顾

1. **加载期重打包**：GGUF 权重一次重排成 panel（Q4Kx8/Q6Kx8/Q2x8/STQ1_0x8）
   并预解码 meta，热循环只有乘加（见 `04-quantization.md`）。
2. **decode/prefill 分流**：decode 用 GEMV（1 token，激活 Q8_K 一次量化多次点积）；
   prefill 用 4×8 panel GEMM，激活 `QuantizeQ8Kx4` 一次量化 4 个 token。
3. **共享激活量化**：`MulMatPanel` / `MulMatQ2` / `MulMatSTQ` 一个并行区段量化一份
   Q8 面板，连续跑最多 3 个矩阵（gate/up，或 q/k/v），ffn_down 用
   `SiluQuantizeAndGemm` 融合 SiLU。
4. **L2 列分块**：`GemmQ4K.ColTileBytes=128KB`（Q6 翻倍、Q2 64KB），权重切片驻留
   L2，q8 行流式扫描。
5. **线程拓扑**：物理 P-core 数工作线程 + 亲和性钉核 + job-id 轮询 + 自旋 barrier，
   见 `05-kernels.md`。
6. **标量回退**：每个内核都有非 AVX2 路径，保证正确（慢）。

## 11.2 剖析钩子（`Qwen35Model` 静态计数器）

`ProfileEnabled` 置 1 时，decode/attention 路径用 `Stopwatch.GetTimestamp()` 包裹
各算子并累加到静态计数器：

```
TicksQ4 / TicksQ2 / TicksSTQ / TicksQ6   （各类权重的 GEMM/GEMV 耗时）
TicksAttnScore / TicksSoftmax / TicksAttnCombine
TicksRms / TicksRope / TicksSilu / TicksEmbed / TicksDelta / TicksConv
```

`ResetProfile()` 清零。把几个周期累加后除以 `Stopwatch.Frequency` 即可得到算子级
耗时分布。注意部分计数器只在 decode 路径累积（如 `TicksRope`、`TicksDelta`），
prefill 路径还未全部插桩。

## 11.3 诊断环境变量

| 变量 | 值 | 输出 |
|------|-----|------|
| `Q35_TRACE` | `1` | 每层每个子步骤（attn_norm、gemm q、conv、recurrence、moe、ffn_down…）进入/完成的 stderr 跟踪 |
| `Q35_DIAG` | `1` | 每 3 层打印最后 token 隐藏态 RMS 范数 `|h_last|`；前向末尾打印整体 `|h|` / `|normed|` |

用于快速定位“哪一层开始数值发散”与层内热点。

## 11.4 内存策略（前向期）

* 权重：GGUF mmap 后**整块拷入** 64B 对齐 `NativeBuffer`；panel 额外拷贝（Q4_K
  约 8× + meta、Q6 约 13×、Q2 约 8×、STQ 约 13×）——一次性只读。
* 激活：模型级 `_scratch` 大 arena，`Bump` 64B 对齐、层间回卷、按实际 `seq`/
  `kvLen` 懒增长，避免为配置上下文预分配 GB 级内存。
* KV 缓存：只给全注意力层分配，`[min(ctx,4096)][kvStride]` 起步，倍增扩容。
* 线性层状态：固定大小 `[heads][kDim][vDim]` 与卷积窗口，与序列长度无关。

## 11.5 线程与拓扑

* `CpuThreadPool` 默认按 `CpuTopology.PreferPCoreCount`（物理 P-core 数）建线程，
  上限 `logical`。顺序算子（RMS/SiLU）会迅速回落到休眠，避免烧 P-core。
* Windows 上把 worker 钉到 P-core 逻辑 id；亲和性 best-effort。
* VM 检测避免每个物理核多一个 worker（lockstep GEMM 吞吐崩溃）。

## 11.6 已知瓶颈与优化方向

1. **MoE prefill**：逐 token 逐专家 GEMV（每次独立 Q8 量化），未批处理。方向：
   同一专家被多个 token 选中时合并成 panel GEMM。
2. **Gated-DeltaNet 递推**：纯顺序、每个 token 分配托管数组（`float[] qN/kN`）
   做 L2 归一化，且 `DeltaStep` 内部有多次 pass。方向：栈分配/就地归一、
   向量化 `kvMem`/write/read 循环。
3. **prefill 部分路径未插桩**：剖析主要覆盖 decode；prefill 的 GEMM 已用
   `TicksQ4` 等统计，但 conv/delta/recurrence 的 prefill 版未单独计。
4. **Q5_K / IQ4_XS / STQ1_0 prefill**：退化为逐 token GEMV，未做 panel GEMM。
5. **softmax 的 `FastExp`**：每元素一条分支；批量路径已向量化。
6. **KV 缓存是 f32**：未做 KV 量化（f16/i8）——f32 简单但 decode 带宽较高。