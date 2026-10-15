# 5. 内核与运行时（`Qwen35Sharp.Kernels`）

## 5.1 GEMM/GEMV 内核总览

| 内核 | 用途 | 路径 |
|------|------|------|
| `VecDotQ4K.Dot` | decode 单行 Q4_K × Q8_K 点积 | scalar 版 + `DotAvx2`（net 通道，accMin 用 SSE 低位累积） |
| `Q6K.Dot` / `DotAvx2x4` | decode / 4-token Q6_K 点积 | `DotAvx2`（掩码提升 6-bit）；`DotAvx2x4` 一解多用 |
| `Q2_0C.Dot` | decode Q2_0C 点积 | `Dot256`（字节→dword→独立字节展开） |
| `IQ4XS.Dot` / `Gemv` / `Gemm` | IQ4_XS 非线性 4-bit | 查表（`kvalues_iq4nl`），`Gemv` 内部做 Q8 量化或复用已有 Q8 |
| `STQ1_0.Dot` / `STQPanel.Gemv/Gemm` | STQ1_0 稀疏三元 | `DotChunk16`（vpshufb 码本 + SignMaskLut） |
| `GemmQ4K.Gemm8x8` | prefill 4×8 Q4_K GEMM | scalar（对齐 `ggml_gemm_q4_K_8x8_q8_K_generic`）+ AVX2（零 shuffle 端口） |
| `GemmQ6K.Gemm8x8` | prefill 4×8 Q6_K GEMM | scalar + AVX2（u8 面板，−32 修正） |
| `Q2Panel.Gemv/Gemm` | Q2_0C panel | AVX2 |
| `STQPanel.Gemv/Gemm` | STQ panel | AVX2 |
| `MulMatQ4K` | Q4_K 调度（decode GEMV / prefill GEMM）+ 填充尾列 | 统一入口 |
| `MulMatQ6K` | Q6_K 调度 | 统一入口 |
| `MulMatQ2` / `MulMatSTQ` | Q2/STQ 的 panel + 尾部回退 | 统一入口，支持最多 3 个矩阵共享一份 Q8 |
| `MulMatPanel` | prefill 融合：量化 → barrier → 至多 3 个 Q4/Q6 panel GEMM | `QuantizeAndGemm` / `SiluQuantizeAndGemm` |

### 5.1.1 调度约定

每个 `MulMat*` 的 `Gemm(...)` 入口都遵循：

```
if (tokens == 1) → GEMV 路径（量化激活行为 BlockQ8K，逐行点积）
else             → prefill 路径（4-token 组 → BlockQ8Kx4 → panel GEMM）
```

MQ4_K/MQ6_K 处理 token 数不是 4 的倍数（补零到 4、展平、回拷，见 `06/07`），并处理
`nOut % 8` 尾部（用逐行 GEMV 补算；`GemmQ4K` 面板要求 cols 为 8 的倍数）。

## 5.2 AVX2 技术要点

### 列序修正

`vpmaddubsw`/`vphadd` 通道天然产生 `[0 1 4 5 2 3 6 7]` 的列序。所有 panel 内核
用一次 `Avx2.PermuteVar8x32(acc, ColumnOrder)` 修正后存储（`Q2Panel`/`STQPanel`/
`GemmQ4K`/`GemmQ6K`）。

### 主循环的无端口竞争（Q4_K panel）

`GemmQ4K.Avx2` 对每个激活行用 `vpbroadcastq`（纯加载）对着**原始列交织 nibble**
做 `vpmaddubsw`：每 8 个 `vpmaddubsw` 只有 8 adds、2 ands、6 loads，没有 shuffle
端口工作。scales 从 `BlockQ4Kx8Meta` 预解码（见上节）。

### int16 溢出守卫

* Q4_K：每 short ≤ 8·15·127，行列 pair 和可安全放进 int16
  （`RowSum` 先 widen 再 reduce）。
* Q2：每 short 含 64 个乘积，|sum|≤64·3·128=24576；`RowSum`/`TotalSums` 在
  **int16 横向加之前**先 widen 到 int32（注释强调 uniform 极端输入下 int16 HADD
  会溢出）。
* Q6：每 short ≤ 4·63·127。

### Q6_K 的展开技巧

预fill 面板把 Q6 值展开成 u8 0..63（`RepackQ6K.Expand`），点积输出累计为
`Σ(sc·(q-32)·a)`。内核用 `ScalePairs` 与 Q8 的 `Bsums` 一次算出 −32 修正：
`jcc -= 32 · Σ(sc_i · bsum_i)`，每 super-block 一次，把 −32 折叠进整数累加器
（`ShiftLeftLogical(c, 5)` = ×32 减）。

### 软 exp/silu（`FastExp`）

对 llama.cpp `ggml_v_expf`/`ggml_v_silu` 的精确移植（`vec.h`）：按 2^(n) 分解 +
f32 多项式，max ~1.45 ulp；`|x|>88.38` 冲 +inf，`< -103.97` 冲 0；大输入、负输入
分支覆盖。`SiluAvx2 = x / (1 + exp(−x))`。被 softmax、SiLU、conv 的 SiLU、FFN 门控
共用。

### 标量回退

每个内核都提供无 AVX2 的标量路径（`DotScalarForTest`、`GemmScalar`、
`Quantize4x8Scalar` 等），保证在旧 CPU 上仍正确（慢）。热路径一般要求
`Avx2.IsSupported && Fma.IsSupported`。

## 5.3 线程池（`CpuThreadPool`）

* 构造：`threadCount<=0` 时用 `CpuTopology.PreferPCoreCount`；生成
  `ThreadCount = max(1, …)` 个后台线程，`AboveNormal` 优先级。
* 派发：`For(count, body)` 把 `[worker=0..workers)` 按行写入 `_work`，`volatile`
  `_workers/_remaining`，`Interlocked.Increment(ref _jobId)`，唤醒相关线程，主线程
  自旋等待 `_remaining` 归零。
* 工作线程：轮询 `_jobId` 变化 → 执行 `_work(worker, workers)` → `Decrement(_remaining)`；
  `jobId` 不变时自旋约 8192 次后 `AutoResetEvent.WaitOne()` 休眠（顺序化
  RMS/SiLU 不烧 P-core）。
* `Barrier()`：ggml 式自旋 barrier（`_barrierCount` 累积到 `_workers` 即翻
  `_barrierGen`），用于同一 `For` 内“先量化后 GEMM”的同步。
* `TryPinToPCore`：Windows 上按拓扑把每个 worker `SetThreadAffinityMask` 钉到
  P-core 逻辑 id（best-effort，失败静默）。
* `Dispose`：置 `_stop`、翻 `jobId`、唤醒所有线程并 `Join`。

## 5.4 CPU 拓扑（`CpuTopology`）

静态初始化通过操作系统接口发现**物理核心**：

* Windows：`GetLogicalProcessorInformationEx(RelationProcessorCore)` 解析效率类与
  逻辑 id（group 感知）。
* Linux：`/sys/devices/system/cpu/{present,online}` + `cpu_core/cpus` +
  `thread_siblings_list`。

`ChooseWorkerCount` 的规则：

1. 无拓扑 → VM 见偶数 vCPU 减半，否则 `min(4,..)`/`÷2` 估算。
2. 有拓扑 → 默认取**物理 P-core** 数；若 VM 隐藏 SMT（每 vCPU 看着像独立核心），
   减半避免每物理核多一个 worker 拖垮 lockstep GEMM。

P-core 判定：Intel hybrid 上 SMT 只在 P-core 出现（有 SMT/无 SMT 并存 → SMT 的就是
P）；否则取效率类最低的一批。`AutoHint` 解释选择原因，CLI 打印给用户看。
VM 检测用 CPUID hypervisor 位 + 厂商字符串 + Windows 注册表/固件名。

## 5.5 内存原语（`NativeBuffer` / `ScratchArena`）

* `NativeBuffer`：`NativeMemory.AlignedAlloc(64)` + 清零 + 终结器兜底；`Span` 视图
  提供托管侧安全访问。
* `ScratchArena`：6 个惰性扩容的 `NativeBuffer`（A..F），不回收复用——调用方覆盖其
  需要的字节。模型层按此约定把每层不同用途的中间量分到不同槽位，避免互相踩踏。

## 5.6 面板 GEMM 的 L2 分块

`GemmQ4K.ColTileBytes = 128 KB`。`GemmAvx2` 把列方向（输出维度）切成 tile：权重
切片留在 L2，q8 行流式扫过，避免反复从 RAM 拉权重。Q6 用 Q4 的 2 倍 tile；
Q2 面板 `MulMatQ2.ColTileBytes = 64 KB`，按 8 列一组 tile 化。

## 5.7 无 AVX2 时的退化路径

`MulMatQ4K.Gemv` 在无 AVX2 时使用 `VecDotQ4K.DotScalar`；`MulMatQ2`/`MulMatSTQ`
在没有 AVX2 或没有 panel 时抛 `PlatformNotSupportedException` 或走原始行点积回退；
`QuantizeQ8Kx4` 提供标量版。模型前向内 `Linear` 按 `w.Type` 分派到对应的
`MulMat*`，因此权重格式决定内核选择，CPU 能力决定 SIMD 宽度选择。