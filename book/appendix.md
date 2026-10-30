# 附录

> 附录是把正文里散落的"抄写件"集中收拢的参考章节。它们不是阅读正文的必需品，但当你动笔实现或改动代码时，它们是你不必重新翻 17 章的速查表。
>
> - 附录 A 内核速查表
> - 附录 B 张量与元数据字典
> - 附录 C 测试索引
> - 附录 D 阅读路径地图

---

## 附录 A 内核速查表

> 锚点源码：`src/Qwen35Sharp.Kernels/BlockLayout.cs`（`Qk`）、`GemmQ4K.Avx2.cs`（`ColTileBytes`）、`MulMatQ2.cs`（`ColTileBytes`）、`doc/04-quantization.md` §4.1、`doc/05-kernels.md`

### A.1 块尺寸常量（`Qk` 静态类）

| 常量 | 值 | 含义 |
|------|----|------|
| `Qk.SuperBlock` | 256 | K 系 super-block 值数 |
| `Qk.ScaleBytes` | 12 | Q4_K/Q5_K 的 scales 字节数 |
| `Qk.Q4KSize` | 144 | Q4_K 块字节数 |
| `Qk.Q8KSize` | 292 | Q8_K 块字节数 |
| `Qk.Q4Kx8Size` | 1152 | 8 列 Q4_K panel 块 |
| `Qk.Q4Kx8MetaSize` | 448 | Q4_K panel 的预解码 meta |
| `Qk.Q8Kx4Size` | 1168 | 4 行 Q8_K panel 块 |
| `Qk.Q5KSize` | 176 | Q5_K 块 |
| `Qk.Q6KSize` | 210 | Q6_K 块 |
| `Qk.IQ4XSSize` | 136 | IQ4_XS 块 |
| `Qk.Q6Kx8Size` | 2848 | 8 列 Q6_K panel |
| `Qk.Q2_0CSize` | 130 | Q2_0C 块（512 值） |
| `Qk.Q2x8Size` | 1056 | 8 列 Q2 panel |
| `Qk.STQ1_0BlockLength` | 256 | STQ1_0 块值数 |
| `Qk.STQ1_0Size` | 42 | STQ1_0 块字节 |
| `Qk.STQ1_0x8Size` | 544 | 8 列 STQ panel（8·4 + 512） |

### A.2 各格式内核一览

| 内核 | 用途 | 关键实现 |
|------|------|---------|
| `VecDotQ4K.Dot` | decode 单行 Q4_K × Q8_K | `DotAvx2` 用 `vpmaddubsw` + SSE 低位槽累积 accMin |
| `Q6K.Dot` | decode Q6_K 点积 | `DotAvx2` 用 m3/m12/m48/mC0 掩码提升 6-bit；`DotAvx2x4` 一解多用（4 行） |
| `Q2_0C.Dot` | decode Q2_0C 点积 | `Dot256` 字节→dword→独立字节展开，无临时权重存储 |
| `IQ4XS.Dot` | decode IQ4_XS 非线性 4-bit | 查 `kvalues_iq4nl` 表，免展开 |
| `STQ1_0.Dot` | decode STQ 稀疏三元 | `DotChunk16` + `CodebookLo/CodebookHi` + `SignMaskLut` 查表 |
| `GemmQ4K.Gemm8x8` | prefill 4×8 Q4_K GEMM | `GemmAvx2` 无 shuffle 端口的 `vpmaddubsw` 结构 |
| `GemmQ6K.Gemm8x8` | prefill 4×8 Q6_K GEMM | u8 展开面板 + −32 修正（`ShiftLeftLogical(c,5)`） |
| `Q2Panel.Gemv/Gemm` | Q2_0C panel | AVX2，一次广播喂 4 列 |
| `STQPanel.Gemv/Gemm` | STQ panel | AVX2 |
| `MulMatQ4K` | Q4_K 调度 + 尾部补算 | `tokens==1` → GEMV，否则 → GEMM；`PackedGroups(nOut)=nOut/8` |
| `MulMatPanel` | prefill 融合 | `QuantizeAndGemm`（至多 3 个 panel 共享一份 Q8）；`SiluQuantizeAndGemm` 融合 |

### A.3 调参值

| 参数 | 值 | 位置 |
|------|----|------|
| `GemmQ4K.ColTileBytes` | 128 KB | `GemmQ4K.Avx2.cs` |
| `GemmQ6K` 的 tile | 2 × Q4（约 256 KB） | `GemmQ6K.cs` |
| `MulMatQ2.ColTileBytes` | 64 KB | `MulMatQ2.cs` |
| KV 缓存起点 | `min(ContextLength, 4096)`，倍增扩容 | `Qwen35Model.cs`（`EnsureCache`） |
| scratch 余量 | 4 MB | `Qwen35Model.cs`（`EnsureScratch`） |

---

## 附录 B 张量与元数据字典

> 锚点源码：`src/Qwen35Sharp.Model/ModelConfig.cs`（`FromGguf`）、`Qwen35Model.cs`（`LoadWeights`）、`doc/03-gguf.md`、`doc/06-model.md`、`doc/07-forward.md`

### B.1 模型层结构速查（所有层通用）

```
hidden → attn_norm（zero-centered RMSNorm）
       → token 混合器（全注意力 或 Gated-DeltaNet）→ 残差
       → post_attention_norm（zero-centered RMSNorm）→ FFN → 残差
```

### B.2 全注意力层张量（`blk.L.*`，`IsRecurrent == false`）

| 张量 | 形状 | 用途 |
|------|------|------|
| `attn_norm.weight` | hidden | 层输入 RMSNorm |
| `attn_q.weight` | hidden → 2·qDim | 查询 + 门控（2 倍宽，头内交错） |
| `attn_k.weight` | hidden → kvDim | 键（GQA） |
| `attn_v.weight` | hidden → kvDim | 值 |
| `attn_q_norm.weight` / `attn_k_norm.weight` | headDim | per-head RMSNorm |
| `attn_output.weight` | qDim → hidden | 输出投影 |
| `post_attention_norm.weight` | hidden | 残差后 norm |
| `ffn_gate/up/down.weight` | hidden ↔ ffn | SiLU-MLP 三段 |

### B.3 Gated-DeltaNet 层张量（`blk.L.*`，`IsRecurrent == true`）

| 张量 | 形状 | 用途 |
|------|------|------|
| `attn_qkv.weight` | hidden → convDim | 拼接 q/k/v |
| `attn_gate.weight` | hidden → valueDim | 门控 z |
| `ssm_conv1d.weight` | [kernel][convDim] | 因果深度卷积核 |
| `ssm_beta.weight` | hidden → heads | β |
| `ssm_alpha.weight` | hidden → heads | α |
| `ssm_a`（F32） | heads | log A，已烘烤成 −exp(A) |
| `ssm_dt.bias`（F32） | heads | dt 偏置 |
| `ssm_norm.weight` | valueDim | 门控 RMSNorm |
| `ssm_out.weight` | valueDim → hidden | 输出投影 |

### B.4 MoE 张量（`blk.L.*`，arch `qwen35moe`）

| 张量 | 形状 | 用途 |
|------|------|------|
| `ffn_gate_inp.weight` | hidden → experts（F32） | router |
| `ffn_gate_exps.weight` | hidden → moeFfn → experts | 专家 gate（3D） |
| `ffn_up_exps.weight` | hidden → moeFfn → experts | 专家 up（3D） |
| `ffn_down_exps.weight` | moeFfn → hidden → experts | 专家 down（3D） |
| `ffn_gate_inp_shexp.weight` | hidden → 1（F32） | 共享门控 logit |
| `ffn_gate_shexp.weight` / `up_shexp` / `down_shexp` | hidden ↔ sharedFfn | 共享专家 |

### B.5 派生维度（`ModelConfig`）

```
LinearConvDim  = 2·SsmGroupCount·SsmStateSize + SsmInnerSize
LinearKeyDim   =   SsmGroupCount·SsmStateSize
LinearValueDim =   SsmTimeStepRank·SsmStateSize
FullQDim       = NumHeads·HeadDim
FullKvDim      = NumKvHeads·HeadDim
```

### B.6 层类型判定（关键）

```
blk.L.attn_qkv.weight 存在 → 线性注意力层
blk.L.attn_q.weight   存在 → 全注意力层
都没有                   → 回退公式 (L+1) % full_attention_interval != 0
整个文件都没有层张量      → 全部用公式
```

---

## 附录 C 测试索引

> 锚点源码：`tests/Qwen35Sharp.Tests/`、`doc/10-testing.md`

| 测试类 | 用例 | 守护的代码 | 容差/断言 |
|--------|------|-----------|----------|
| `FastExpTests` | `ExpAvx2_TracksMathF` | `FastExp.ExpAvx2` vs `MathF.Exp` | 相对误差 < 2e-5 |
| `FastExpTests` | `SiLUMul_TracksScalar` | `Ops.SiLUMulRange` vs 标量 `x/(1+e^-x)·up` | < 2e-4 |
| `FastExpTests` | `Quantize4x8Silu_MatchesSiLUThenQuantize` | `Quantize4x8Silu` vs "先 SiLU 再量化" | 逐字节一致 |
| `FastExpTests` | `ThreadPool_ForSplitsWork` | `CpuThreadPool.For` 覆盖 [0,32) 每项一次 | 逐项核对 |
| `GatedDeltaNetOpsTests` | `RecurrentGatedDeltaRule_MatchesReference` | `RecurrentGatedDeltaRule`/`DeltaStep` vs 独立标量参考 | < 1e-3（含状态 S） |
| `GgufTypeTests` | `LegacyStride16Type42IsNormalizedWithoutBreakingQ2` | `CanonicalTensorType` 两条分支 | STQ1_0 / Q2_0 各测 |
| `IQ4XSTests` | `DequantizeRow_MatchesReference` | `IQ4XS.DequantizeRow` vs llama.cpp 公式参考 | < 1e-4 |
| `IQ4XSTests` | `Dot_MatchesScalarDequant` | `IQ4XS.Dot` vs 反量化后点积 | 1e-2 / 1e-4·\|ref\| |
| `KvCacheAlignTests` | `DivergingTail_KeepsSharedPrefix` 等 5 例 | `KvCacheAlign.Plan` 各分支 | 行为断言 |

> 已知盲区（doc/10-testing.md §10.4）：无端到端生成测试；MoE/STQ/Q2/Q5/Q6 的 panel 与 GEMV 尚无专项对拍；BPE 与聊天模板无专项单测。

---

## 附录 D 阅读路径地图

> 这是一条"代码徒步"路线，从用户敲下命令到 logits 落地，把 17 章的锚点穿成一条可走的链。沿真实调用链标注文件与类，方便读者"改一处 → 知道会撞到哪"。

```
用户输入
  │
  ▼
src/Qwen35Sharp.Cli/Program.cs
  ├─ Args.Get / Args.GetInt                    （第 14 章）
  ├─ Reply(model, messages, user, maxTokens)
  ├─ ChatTemplate.RenderQwen35(messages)        （第 13 章）
  ├─ model.Tokenizer.Encode(rendered)           （第 13 章，BpeTokenizer.Encode）
  ├─ model.AlignPrompt(promptIds)               （第 10 章，KvCacheAlign + TruncateCache）
  ├─ model.Forward(align.Suffix)                （第 9 章）
  ├─ ArgMax(logits)  → 循环                        （第 14 章）
  ├─ model.Forward([token])                     （第 9 章，seq==1 → DecodeBlock）
  └─ WriteDelta(model, generated, ref visible)  （第 14 章）
        │
        ▼
src/Qwen35Sharp.Model/Qwen35Model.cs
  ├─ Forward(int[] tokens)                      （第 9.1 节）
  │    ├─ EnsureCache / EnsureScratch           （第 8.4 节）
  │    ├─ Embed(tokens, h)                      （第 9.2 节）
  │    ├─ seq==1 ? DecodeBlock : PrefillBlock   （第 9.4 节）
  │    │     ├─ Rms(attn_norm)                  （Qwen35Ops.RmsNormZeroCentered，第 8.3/9.3 节）
  │    │     ├─ 全注意力 → Decode/PrefillAttention        （第 10 章）
  │    │     │     ├─ Linear / DecodeLinear     （第 9.2 节，MulMat* 路由）
  │    │     │     ├─ Ops.NeoXRoPE              （第 10.2 节）
  │    │     │     ├─ Ops.AttentionScores / SoftmaxCausal / AttentionCombine（第 10.3 节）
  │    │     │     └─ Sigmoid 门控 + attn_output （第 10.4 节）
  │    │     ├─ Gated-DeltaNet → Decode/PrefillLinearAttn（第 11 章）
  │    │     │     ├─ Linear(attn_qkv / attn_gate)
  │    │     │     ├─ GatedDeltaNetOps.CausalConv1d / CausalConv1dUpdate（第 11.3 节）
  │    │     │     ├─ GatedDeltaNetOps.RecurrentGatedDeltaRule / DeltaStep（第 11.5 节）
  │    │     │     └─ Qwen35Ops.GatedRmsNorm + Linear(ssm_out)（第 11.6 节）
  │    │     └─ FFN：DecodeGateUp / SiluQuantizeAndGemm（dense，第 9.4/6.5 节）
  │    │              或 Prefill/DecodeSparseMoe（MoE，第 12 章）
  │    └─ Rms(output_norm) + Linear(最后一行, "output.weight")
  │
  ├─ AlignPrompt（KvCacheAlign.Plan，第 10.5 节）
  ├─ TruncateCache / ResetCache                 （第 10.5/14.3 节）
  └─ 静态 Ticks* 计数器（剖析，第 16.1 节）
        │
        ▼
src/Qwen35Sharp.Kernels/
  ├─ Q8K.QuantizeRow（第 5.2 节）→ 一次量化
  ├─ VecDotQ4K / Q6K / Q2_0C / IQ4XS / STQ1_0 / Q5K 的点积（第 5.3–5.4 节）
  ├─ MulMat*：tokens==1 → GEMV，否则 → panel GEMM（第 6.6 节）
  ├─ QuantizeQ8Kx4.Quantize4x8 / Quantize4x8Silu（第 6.3 节）
  ├─ GemmQ4K / GemmQ6K / Q2Panel / STQPanel（第 6.4 节）
  ├─ MulMatPanel.QuantizeAndGemm / SiluQuantizeAndGemm（第 6.5 节）
  ├─ FastExp.ExpAvx2 / SiluAvx2（第 5.5 节）
  ├─ CpuThreadPool.For / Barrier（第 7.2 节）
  └─ CpuTopology / NativeBuffer / ScratchArena（第 7.1/7.3 节）
        │
        ▼
src/Qwen35Sharp.Gguf/
  ├─ GgufFile：Parse / Get* / Coerce* / CanonicalTensorType（第 3.2–3.6 节）
  ├─ GgufTensorInfo / GgmlTensorType（第 3.5 节）
  └─ 内存映射（第 3.3 节）
```

**徒步守则**：

1. 从 `Program.cs` 往下走：任何"行为问题"（输出错/慢/崩）先确认停在哪一环。
2. 改一个算子：先找它所在的静态类 + 对应的 `Ticks*` 计数器 + 参考对拍测试，三位一体地改。
3. 改格式：先 `BlockLayout.cs` 的 `Qk` 常量 + block struct，再排版/重打包器，再 `MulMat*` 新分支，最后格式测试。
4. 任何改动跑一遍 `dotnet test tests/Qwen35Sharp.Tests/Qwen35Sharp.Tests.csproj`——附录 C 的索引会告诉你哪些测试在守这道门。