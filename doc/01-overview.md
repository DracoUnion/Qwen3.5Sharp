# 1. 项目概述

## 1.1 项目定位

Qwen35Sharp 是一个纯 CPU 的 C# / .NET 10 推理引擎，直接从一个 `.gguf` 权重文件
加载 **Qwen3.5** 系列模型并执行自回归文本生成。它不依赖任何深度学习框架，也不依赖
GPU/原生库；全部计算由托管代码中的 AVX2/FMA/AVX-VNNI 向量指令完成。

项目骨架继承自同源的 [HyMT2Sharp](../HyMT2Sharp)：内存映射 GGUF 读取器加上针对
量化权重的高性能 GEMM/GEMV 内核。本项目在其之上实现 Qwen3.5 的**混合** Transformer
结构。

## 1.2 Qwen3.5 架构简述

Qwen3.5 是一种混合（hybrid）Transformer，每隔 N 层（`full_attention_interval`，
Qwen3.5 中为 4；从 1 开始计数，即层 3、7、11、…）插入一层标准 GQA 全注意力层，
其余层为 **Gated-DeltaNet** 线性注意力层（又称 gated SSM）。每层之后跟一个 SiLU
门控 FFN（dense 或 MoE）。

任意一层 `L` 的统一结构：

```
hidden → attn_norm (zero-centered RMSNorm)
       → token mixer（全注意力 或 Gated-DeltaNet）
       → +hidden（残差）
       → post_attention_norm (zero-centered RMSNorm)
       → FFN（dense SiLU-MLP  或 稀疏 MoE）→ +hidden（残差）
```

* **全注意力层**：`attn_q`（输出为 2 倍宽：query ∥ gate）、`attn_k`、`attn_v`、
  `attn_q_norm` / `attn_k_norm`（按头的 RMSNorm）、部分 RoPE、GQA 缩放 softmax，
  然后 `out = out * sigmoid(gate)` 再 `attn_output` 投影。sigmoid 门控是 Qwen3.5
  相对 Qwen2 的一个关键改动。
* **Gated-DeltaNet 层**：`attn_qkv`（拼接的 q/k/v）、`attn_gate`（＝ z）、
  `ssm_conv1d`（因果深度卷积 + SiLU）、`ssm_beta`（＝ β）、`ssm_alpha`（＝ α，
  配合 `ssm_dt.bias` → softplus）、`ssm_a`（＝ log A → −exp(A)）、门控 RMSNorm
  （`ssm_norm`）、`ssm_out`。delta-rule 递推在序列上（prefill）或单 token（decode）
  运行，更新每层 `[heads][k_dim][v_dim]` 状态缓冲；卷积状态是一个滚动
  `[k−1][conv_dim]` 窗口。

## 1.3 设计目标

1. **正确性优先**：核心递推（Gated-DeltaNet delta rule）用与参考实现对拍的单测
   锁定（见 `GatedDeltaNetOpsTests`）。
2. **纯托管 + SIMD**：不依赖 GPU、OpenBLAS/MKL 等原生库；底层用
   `System.Runtime.Intrinsics.X86` 直接生成 AVX2/AVX-VNNI/FMA 指令。
3. **量化和性能**：decode 走量化 GEMV，prefill 走 4×8 panel GEMM，激活量化为
   Q8_K 并跨多个矩阵复用同一份量化结果。
4. **内存可控**：权重只需编码后的字节；激活用 64 字节对齐的 scratch arena 按实际
   序列长度懒分配，避免为配置的上下文窗口预保留数 GB。
5. **无本地状态 / 便于集成**：核心是可复用的 `Qwen35Model` 类，CLI 只是薄壳。

## 1.4 非目标

* **视觉 / 多模态**：不支持 `Qwen3_5VisionModel`、3D 位置 id、M-RoPE 分段。
  `ModelConfig.RopeDim` 是部分旋转的 1 维 RoPE 维度。
* **投机解码 / MTP**：不实现。
* **分块（chunked）Gated-DeltaNet**：只做顺序递推。顺序递推与分块在数学上等价，
  且与 llama.cpp 逐 token decode 一致，对任意长度 prefill 都是正确的；分块只是
  未来的内存优化方向。
* **采样算法多样性**：CLI 目前只做贪心 argmax，不做 top-k/top-p 采样。

## 1.5 技术栈与工程配置

`Directory.Build.props` 全局生效：

| 配置 | 值 | 备注 |
|------|-----|------|
| `TargetFramework` | `net10.0` | 需要 .NET 10 SDK |
| `Nullable` | `enable` | 全库可空注解 |
| `LangVersion` | `latest` | 使用最新 C# 特性 |
| `AllowUnsafeBlocks` | `true` | 内核与张量访问使用不安全指针 |
| `InvariantGlobalization` | `true` | 加速、避免区域差异 |
| `Deterministic` | `true` | 可复现构建 |
| `RootNamespace` / `AssemblyName` | `Flygon.*` | 命名空间前缀 `Flygon` |
| `PackageLicenseExpression` | `Apache-2.0` | 许可 |

项目清单（`Qwen35Sharp.slnx`，.NET 10 的新式 XML/slnx 解决方案文件）：

```
src/Qwen35Sharp.Gguf/Qwen35Sharp.Gguf.csproj     内存映射 GGUF 读取器
src/Qwen35Sharp.Kernels/Qwen35Sharp.Kernels.csproj  量化 GEMM/GEMV 内核 + 线程池
src/Qwen35Sharp.Model/Qwen35Sharp.Model.csproj    模型本体（引用 Kernels + Gguf）
src/Qwen35Sharp.Cli/Qwen35Sharp.Cli.csproj        交互式命令行
tests/Qwen35Sharp.Tests/Qwen35Sharp.Tests.csproj  xunit 测试
```

## 1.6 构建与运行

```bash
dotnet build Qwen35Sharp.slnx
dotnet test  tests/Qwen35Sharp.Tests/Qwen35Sharp.Tests.csproj

dotnet run --project src/Qwen35Sharp.Cli -- ^
  --model "D:\_\model\Qwen3.5-4B-Q4_K_M.gguf" ^
  --prompt "Say hello"
```

若 .NET 10 SDK 装在用户目录：

```bash
$env:DOTNET_ROOT = "$env:USERPROFILE\.dotnet"
dotnet build Qwen35Sharp.slnx
```

省略 `--prompt` 进入交互式 REPL；`/reset` 清空对话与缓存，`/exit` 退出。
CLI 会在启动时打印线程数、自动提示原因、架构与超参、AVX2/VNNI 支持情况。