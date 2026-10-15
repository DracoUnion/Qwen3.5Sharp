# Qwen35Sharp 详细设计文档

本项目为面向 **Qwen3.5** GGUF 模型的 C# / .NET 10 CPU 推理引擎。本目录提供完整
的技术设计与实现说明，涵盖 GGUF 读取、量化内核、模型前向、MoE、分词器、运行时
与测试。

## 章节导航

| 章节 | 内容 |
|------|------|
| [01-overview.md](01-overview.md) | 项目概述、设计目标、非目标、技术栈、目录结构、构建与运行 |
| [02-architecture.md](02-architecture.md) | 分层架构、模块依赖、推理数据流、关键设计决策（D1–D7） |
| [03-gguf.md](03-gguf.md) | GGUF v2/v3 文件格式、内存映射读取器、元数据解析、类型规范化 |
| [04-quantization.md](04-quantization.md) | 支持的量化格式（Q4_K/Q5_K/Q6_K/Q8_K/Q2_0C/STQ1_0/IQ4_XS）、块布局、panel 重打包 |
| [05-kernels.md](05-kernels.md) | 量化 GEMM/GEMV 内核、SIMD 技巧、逐点算子、线程池、CPU 拓扑、内存管理 |
| [06-model.md](06-model.md) | 模型配置、权重加载与 `Weight` 结构、线性分派、Embed |
| [07-forward.md](07-forward.md) | Prefill / Decode 前向、全注意力层、Gated-DeltaNet 线性注意力层、KV 缓存与前缀复用 |
| [08-moe.md](08-moe.md) | MoE 稀疏 FFN（router、top-k、专家 GEMV、共享专家、3D 张量布局） |
| [09-tokenizer-cli.md](09-tokenizer-cli.md) | 字节级 BPE 分词器、聊天模板、交互式 CLI |
| [10-testing.md](10-testing.md) | 测试策略与各测试用例说明 |
| [11-performance.md](11-performance.md) | 性能剖析钩子、诊断环境变量、内存与线程策略 |
| [12-limitations.md](12-limitations.md) | 未实现的功能、已知限制与未来工作 |

## 快速要点

* 复用 [HyMT2Sharp](../HyMT2Sharp) 骨架：内存映射 GGUF 读取器 + AVX2/AVX-VNNI
  量化 GEMM/GEMV 内核。
* Qwen3.5 是**混合** Transformer：每隔四层（从 1 计数）是全 GQA 注意力层（带
  sigmoid 门控），其余为 **Gated-DeltaNet** 线性注意力层；穿插 SiLU-FFN。
* 支持两种架构：`qwen35`（dense SiLU FFN）与 `qwen35moe`（稀疏 MoE FFN +
  可选共享专家）。
* 设计主线：**加载期一次性重打包成 panel 格式，热循环不做位手术**；decode 走
  GEMV，prefill 走 4×8 panel GEMM；激活统一量化为 Q8_K。
* 纯文本引擎：不支持视觉/多模态、投机解码、分块线性注意力。

## 源码地图

```
src/Qwen35Sharp.Gguf/   GgufFile.cs · GgmlTensorType.cs · GgufTensorInfo.cs
src/Qwen35Sharp.Kernels/ BlockLayout.cs · Q2_0C.cs · Q4K.cs · Q5K.cs · Q6K.cs ·
                          Q8K.cs · IQ4XS.cs · STQ1_0.cs · VecDotQ4K.cs ·
                          QuantizeQ8Kx4.cs · RepackQ2.cs · RepackQ4K.cs ·
                          RepackQ6K.cs · RepackSTQ.cs · GemmQ4K.cs ·
                          GemmQ4K.Avx2.cs · GemmQ6K.cs · MulMatQ2.cs ·
                          MulMatQ4K.cs · MulMatQ6K.cs · MulMatSTQ.cs ·
                          MulMatPanel.cs · Q2Panel.cs · STQPanel.cs ·
                          FastExp.cs · CpuThreadPool.cs · CpuTopology.cs ·
                          NativeBuffer.cs · ScratchArena.cs
src/Qwen35Sharp.Model/  ModelConfig.cs · Qwen35Model.cs ·
                          Qwen35Model.Decode.cs · Qwen35Model.Prefill.cs ·
                          Qwen35Model.Moe.cs · Qwen35Ops.cs · Ops.cs ·
                          GatedDeltaNetOps.cs · KvCacheAlign.cs ·
                          BpeTokenizer.cs · ChatTemplate.cs
src/Qwen35Sharp.Cli/    Program.cs
tests/Qwen35Sharp.Tests/ FastExpTests.cs · GatedDeltaNetOpsTests.cs ·
                          GgufTypeTests.cs · IQ4XSTests.cs ·
                          KvCacheAlignTests.cs
```