# 12. 限制与未来工作

## 12.1 尚未实现

| 能力 | 现状 | 说明 |
|------|------|------|
| 视觉 / 多模态 | 不支持 | 无 `Qwen3_5VisionModel`、无 3D 位置 id、无 M-RoPE 分段。`ModelConfig.RopeDim` 是部分旋转的 1 维 RoPE 维度；完整 M-RoPE 分段不在范围内 |
| 投机解码 / MTP | 不支持 | — |
| 分块 Gated-DeltaNet | 只做顺序递推 | 顺序与分块数学等价，与 llama.cpp 逐 token decode 一致，对任意长度 prefill 正确；分块是未来长序列内存优化 |
| 采样 | 仅贪心 argmax | CLI 无 top-k / top-p / 温度采样 |
| KV 量化 | 无 | KV 缓存为 f32；f16/i8 KV 可省带宽（decode 带宽受限） |

## 12.2 已知限制

* **纯 CPU / x86 依赖**：热路径要求 AVX2+FMA；Q2/STQ panel 路径在无 AVX2 时抛
  `PlatformNotSupportedException`（其余格式有标量回退）。无 AArch64/NEON 路径。
* **token_embd / 量化张量 nIn 必须是 256 的倍数**（`LoadWeights` 检查）。
  绝大多数 GGUF 量化权重满足；但 embedding 层若 inner dim 非 256 会拒绝加载。
* **nOut 面板要求**：只有 `nOut >= 8` 且 `nExpert == 1` 的 Q4_K/Q6_K/Q2_0C/STQ1_0
  张量才重打包；其余走 GEMV 尾部或逐行点积。
* **STQ1_0 兼容标记**：早期 HyMT2 Stride16 文件需 raw type 42 + file_type 41 +
  名称含 `Stride16` 才被识别为 STQ 并规范化；否则按标准 Q2_0 解释。
* **前缀缓存截断不影响线性层**：`TruncateCache` 只回滚全注意力 KV；线性层状态在
  对话中只增不减，只有 `/reset`（`ResetCache`）才全清。多轮对话中发生“回退重写”
  时线性层状态可能不与 KV 同步。
* **`EnsureScratch` 用启发式估算**：若 MoE/线性层维度组合超出估算，`Bump` 抛
  `InsufficientMemoryException`（已给出提示信息；需更新估算公式）。
* **MoE `expert_weights_scale`**：已解析但未应用到专家输出。
* **Q5_K prefill**：无 panel，逐 token GEMV，吞吐低于 Q4_K/Q6_K。

## 12.3 未来工作（建议方向）

1. **MoE 批处理**：把“同一专家、多个 token”的 GEMV 合并成 panel GEMM；共享专家
   已用 GEMM。
2. **Gated-DeltaNet 递推优化**：
   * 就地 L2 归一化（免去每次分配 `float[]`）；
   * 向量化 `kvMem`/`write`/`read` 三 pass（现为标量循环）；
   * 可选：分块复现（chunk）以降低长序列 prefill 的数组大小。
3. **更多量化类型**：Q3_K、TQ/TQ2、BF16 等读入支持；Q5_K/IQ4_XS/STQ1_0 的
   prefill panel GEMM。
4. **端到端测试与基准**：加入一个小的公开 GGUF 做生成冒烟 + tok/s 基准，配合
   `Q35_TRACE`/`Q35_DIAG`/剖析计数器做回归。
5. **CLI 采样丰富**：top-k/top-p/温度、`--seed`、KV 状态导出。
6. **KV 量化**：f16/i8 KV 缓存 + 量化反量化内核，降低 decode 带宽。
7. **ARM64 路径**：为 AArch64 .NET 提供 NEON/SVE 内核回退。