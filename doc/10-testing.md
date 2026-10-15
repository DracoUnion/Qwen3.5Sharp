# 10. 测试策略（`Qwen35Sharp.Tests`）

xunit 2.9 + `Microsoft.NET.Test.Sdk`，引用全部三个核心项目。测试重点是**数学等价
性**：把内核/递推与独立标量参考实现对拍，而不是只测返回值是否“合理”。

## 10.1 测试用例清单

### `FastExpTests`
| 用例 | 验证内容 |
|------|----------|
| `ExpAvx2_TracksMathF` | AVX2 快速 exp 与 `MathF.Exp` 相对误差 < 2e-5 |
| `SiLUMul_TracksScalar` | `Ops.SiLUMulRange` 与标量 `x/(1+e^-x)·up` 相对误差 < 2e-4 |
| `Quantize4x8Silu_MatchesSiLUThenQuantize` | 融合的 `Quantize4x8Silu` 与“先 SiLU 再 Quantize4x8”逐字节一致 |
| `ThreadPool_ForSplitsWork` | `CpuThreadPool.For` 的工作划分恰好覆盖 [0,32) 每项一次 |

### `GatedDeltaNetOpsTests`
| 用例 | 验证内容 |
|------|----------|
| `RecurrentGatedDeltaRule_MatchesReference` | delta-rule 递推（`l2normQk=true`）与**独立标量参考实现**（同样的数学，独立代码路径）在随机 q/k/v/g/β 上逐元素误差 < 1e-3，状态 `S` 与输出均对拍 |

这是项目**最重要的测试**：锁定 Gated-DeltaNet 递推的正确性，任何后续重构都必须
继续通过。

### `GgufTypeTests`
| 用例 | 验证内容 |
|------|----------|
| `LegacyStride16Type42IsNormalizedWithoutBreakingQ2` | 手写最小 GGUF：含 `"Stride16"` 名称 + file_type 41 + raw type 42 的文件规范化成 `STQ1_0`；普通 Q2_0 文件保持 `Q2_0`，互不误读 |

### `IQ4XSTests`
| 用例 | 验证内容 |
|------|----------|
| `DequantizeRow_MatchesReference` | 随机字节（修正 fp16 网格尺度为 1.0）下 `IQ4XS.DequantizeRow` 与按 llama.cpp 公式的手写参考逐元素一致（< 1e-4） |
| `Dot_MatchesScalarDequant` | `IQ4XS.Dot` 与“反量化后点积”参考一致（绝对/相对容差 1e-2 / 1e-4·|ref|） |

### `KvCacheAlignTests`
| 用例 | 验证内容 |
|------|----------|
| `DivergingTail_KeepsSharedPrefix` | 分歧 tail 保留公共前缀（截断到 3，suffix=[6,7]） |
| `LongerCacheThanPrompt_DoesNotDropPrefix` | 缓存长于提示词时不丢前缀 |
| `FullHit_ReplaysLastToken` | 全命中重放最后 token（截断到 len-1） |
| `EmptyCache_ForwardsWholePrompt` | 空缓存整段前向 |
| `NoOverlap_Resets` | 无重叠时全部重置 |

## 10.2 测试设计原则

1. **参考对拍**：所有数值测试都实现一个独立、可读的标量/数学参考，SUT 与参考
   在同一随机输入上对比（带相对容差）。避免“用 SUT 验证 SUT”。
2. **确定性随机**：固定种子（1234 / 7 / 9），保证可重现。
3. **极端输入防护**：随机生成的量化块先修正 fp16 尺度，避免 NaN/Inf 干扰断言
   （`ForceValidScales`）。
4. **手写文件构造**：GGUF 测试直接 `BinaryWriter` 手写最小文件，不依赖磁盘真实
   模型。
5. **不依赖真实模型权重**：仓库不含 GGUF 二进制，全部用合成数据，CI 可跑。

## 10.3 运行

```bash
dotnet test tests/Qwen35Sharp.Tests/Qwen35Sharp.Tests.csproj
```

对拍测试需要 AVX2+FMA（`FastExpTests` 断言 `Avx2.IsSupported && Fma.IsSupported`），
在旧 CPU 上 `FastExpTests` 会跳过/失败——这是刻意设定：数值断言针对 SIMD 路径。

## 10.4 覆盖盲区（已知）

* 无端到端生成测试（需要真实模型权重，仓库内未包含）。
* MoE、STQ1_0、Q2_0C、Q5_K、Q6_K 的 panel 与 GEMV 尚无专项对拍测试（Q2_0C 的
  `DotScalarForTest`、STQ1_0 的 `DotScalar` 可作为参考锚点）。
* 分词器（BPE）与聊天模板没有专项单测。