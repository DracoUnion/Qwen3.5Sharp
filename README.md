# Qwen35Sharp

面向 **Qwen3.5** GGUF 模型的 C# / .NET 10 CPU 推理引擎，复用了
[HyMT2Sharp](../HyMT2Sharp) 的骨架：内存映射的 GGUF 读取器，加上
AVX2/AVX-VNNI 量化的 GEMM/GEMV 算子内核。

Qwen3.5 是一种**混合** Transformer：每隔四层（从 1 开始计数，即第 3、7、11、
…… 层）是带 sigmoid 注意力门控的标准 GQA 注意力层，其余层是
**Gated-DeltaNet** 线性注意力层（又称 gated SSM）。本项目实现了这两种层，
外加穿插其中的 SiLU-FFN，全部从一个 `.gguf` 文件读取。

## 目录结构

| 项目 | 用途 |
|---------|---------|
| `Qwen35Sharp.Gguf` | 内存映射的 GGUF v2/v3 读取器（张量 + 元数据）。 |
| `Qwen35Sharp.Kernels` | 量化 `x @ Wᵀ` GEMM/GEMV 内核（Q4_K、Q2_0C、STQ1_0、Q5_K、Q6_K）、SIMD 逐点算子、`CpuThreadPool`。 |
| `Qwen35Sharp.Model` | Qwen3.5 模型本体：权重加载、KV 缓存与循环状态缓存、prefill 与 decode 前向计算、BPE 分词器、聊天模板。 |
| `Qwen35Sharp.Cli` | 交互式命令行程序：对提示词做分词、prefill + decode 循环、流式输出回答。 |
| `Qwen35Sharp.Tests` | xunit 测试，包含与参考实现对拍的 Gated-DeltaNet 递推测试。 |

## 构建

需要 .NET 10 SDK（项目目标框架为 `net10.0`）。

```
dotnet build Qwen35Sharp.slnx
dotnet test  tests/Qwen35Sharp.Tests/Qwen35Sharp.Tests.csproj
```

如果机器上的 .NET 10 SDK 装在用户目录而非 `C:\Program Files`，请把 MSBuild
指向它：

```
$env:DOTNET_ROOT = "$env:USERPROFILE\.dotnet"
dotnet build Qwen35Sharp.slnx
```

## 运行

```
dotnet run --project src/Qwen35Sharp.Cli -- ^
  --model "D:\_\model\Qwen3.5-4B-Q4_K_M.gguf" ^
  --prompt "Say hello"
```

省略 `--prompt` 时会进入交互式 REPL；`/reset` 清空对话，`/exit` 退出。

## Qwen3.5 架构说明

从 `general.architecture = "qwen35"` 读取：

* `qwen35.embedding_length`         → 隐藏层大小
* `qwen35.attention.head_count`    → 全注意力头的数量
* `qwen35.attention.head_count_kv` → GQA 的 KV 头数量
* `qwen35.block_count`             → 层数
* `qwen35.full_attention_interval` → 每隔 N 层为一层全注意力
  （默认 4）；其余层为 Gated-DeltaNet。
* `qwen35.ssm.conv_kernel` / `inner_size` / `state_size` /
  `time_step_rank` / `group_count` → Gated-DeltaNet 的维度。
* `qwen35.rope.*`                  → RoPE 基频 / 旋转维度。
* `qwen35.attention.layer_norm_rms_epsilon` → RMSNorm 的 ε。

每一层：

* **所有层**：`blk.L.attn_norm` → token 混合器 → 残差 →
  `blk.L.post_attention_norm` → SiLU MLP → 残差。
* **全注意力层**：`attn_q`（输出为 2 倍：query ∥ gate）、`attn_k`、
  `attn_v`、`attn_q_norm`、`attn_k_norm`（按头的 RMSNorm）、部分
  RoPE、GQA 缩放 softmax，然后 `out = out * sigmoid(gate)` 与 `attn_output`。
* **Gated-DeltaNet 层**：`attn_qkv`、`attn_gate`、`ssm_conv1d`（因果
  深度卷积 + SiLU）、`ssm_beta`（= β）、`ssm_alpha`（= α，配合
  `ssm_dt` 偏置 → softplus）、`ssm_a`（= log A，→ −exp(A)）、门控
  RMSNorm（`ssm_norm`）、`ssm_out`。delta-rule 递推在序列上运行
  （prefill）或对单个 token 运行（decode），更新每层的
  `[heads][k_dim][v_dim]` 状态缓冲；卷积状态是一个滚动
  `[k-1][conv_dim]` 窗口。

`output.weight`（或共享的 `token_embd.weight`）产生最终 logits。

## 尚未实现

* **视觉 / 多模态**（`Qwen3_5VisionModel`、3D 位置 id、M-RoPE 分段）——
  这是一个纯文本引擎。`ModelConfig` 中的 1 维 `RopeDim` 是部分旋转的维度；
  完整的 M-RoPE 分段处理不在范围内。
* **MoE** FFN —— Qwen3.5 dense 使用标准的 SiLU FFN。
* **投机解码 / MTP**。
* 分块的 Gated-DeltaNet 路径 —— 只使用顺序递推（两者等价，与
  llama.cpp 的逐 token decode 一致，对任意长度的 prefill 都是正确的）。

## 许可证

Apache-2.0。
