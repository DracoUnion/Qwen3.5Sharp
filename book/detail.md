# 《从零自制大模型推理引擎》—— 图书细纲

> 蓝本项目：`Qwen35Sharp`（位于 `D:\src\Qwen3.5Sharp`），一个纯 CPU 的 C#/.NET 10 大模型推理引擎。
>
> 写作立场：**本书不罗列源码**。正文从头到尾跟着一位开发者的真实推进顺序走——先定义目标，再读懂模型文件，一块一块造出计算内核，最后拼装成能对话的程序；源码只在细纲里作为"锚点引用"（项目文件、类名、函数名），用来告诉读者每一步对应工程的哪个角落，读者可按图索骥去读。每一节列出的 "本节要写" 是一段话，说明那一节在开发顺序里承担什么任务、讲清楚什么、验证什么。

---

## 目录（一章一阶段）

本书把从零构建推理引擎的过程切成 **七个阶段、十七个章节**，严格按开发顺序排列：先理解模型 → 再读文件 → 再造计算 → 再拼模型 → 再做界面 → 再用测试与性能收尾。每个阶段结束时，程序都比上一阶段"多跑通一层"。

- **第一阶段 认知与目标**：第 1 章（为什么做、做什么）
- **第二阶段 读懂模型**：第 2 章（模型架构）、第 3 章（文件格式）、第 4 章（量化格式）
- **第三阶段 计算内核**：第 5 章（点积与 GEMV）、第 6 章（面板 GEMM）、第 7 章（线程与内存）
- **第四阶段 拼装模型**：第 8 章（配置与加载）、第 9 章（前向骨架）、第 10 章（全注意力层）、第 11 章（Gated-DeltaNet 线性注意力）、第 12 章（MoE）
- **第五阶段 用户可见的世界**：第 13 章（分词器与模板）、第 14 章（CLI 与生成循环）
- **第六阶段 工程质量**：第 15 章（测试）、第 16 章（性能剖析）
- **第七阶段 边界与远方**：第 17 章（限制与未来）

---

# 第一阶段 认知与目标

## 1. 缘起：为什么要用 C# 从零写一个推理引擎

**本阶段意义**：任何工程书的第一章都是"想清楚再动手"。这一章确立了整本书的范围——纯 CPU、无框架依赖、能本地跑通 Qwen3.5，并铺垫贯穿全书的"主线"：让一段文本从 CLI 走进去，转成 token，过一遍矩阵运算，最后吐回一段文本。

### 1.1 目标与边界 | 参考文献 `README.md`、`doc/01-overview.md`
本节要写：先定义"从零自制"到底意味着放弃什么（不碰 PyTorch、不碰 GPU、不碰 llama.cpp 的代码，只复用它定义的公式与文件格式），再把"最小成功标准"讲清楚——加载一个 `.gguf` 权重文件、跑通一次自回归生成、得到与主流实现一致的结果。顺带说明本书非线性地跳过什么：视觉、投机解码这些都不在射程内。

### 1.2 本书的"主角"：Qwen3.5 混合 Transformer | 参考文献 `README.md`（"Qwen3.5 架构说明"节）、`doc/01-overview.md` §1.2
本节要写：用一个无公式的小节把模型结构讲成一句话——"每四层混一层标准注意力，其余层用 Gated-DeltaNet 线性注意力"——让读者在动手前脑子里先有画面，理解为什么这是本书难得的地方：它不是又一个 Llama 复刻，而是混合架构（标准 GQA 注意力 + gated SSM）的完整实现。

### 1.3 技术选型与工程骨架 | 参考文献 `Directory.Build.props`、`Qwen35Sharp.slnx`、`doc/02-architecture.md` §2.1–§2.2
本节要写：解释为什么选 .NET 10 + AVX2/FMA/AVX-VNNI 手写向量指令、为什么允许 unsafe 指针、为什么全局 `InvariantGlobalization`；然后给出五层的项目结构（Gguf / Kernels / Model / Cli / Tests）以及严格的单向依赖方向 Cli→Model→{Kernels,Gguf}，说明这种分层在工程上意味着什么——内核层不知道模型层存在，模型层不知道命令行存在。

### 1.4 主线：从字节到对话，一次完整的推理旅途 | 参考文献 `doc/02-architecture.md` §2.3
本节要写：不写代码，只用一张数据流图（用户文本→分词→token id→Embed→逐层前向→logits→ArgMax→下一个 token 循环）把这整本书的旅程预览一遍，并指出这条主线上有四个"第一次循环"：文件读取、量化计算、分层前向、自回归生成——后文的每一章都是在为这四个环扣上齿轮。

---

# 第二阶段 读懂模型

## 2. 认识你的目标：Qwen3.5 的数学结构

**本阶段意义**：在写任何代码前，必须先能用数学语言描述模型。这一章节不见一行 C#，但它是全书所有内核选型的依据。开发顺序上这属于"照着规格说明书设计"，规格说明书就是模型的公式与张量命名。

### 2.1 统一的一层：残差、RMSNorm 与 FFN | 参考文献 `doc/01-overview.md` §1.2、`doc/07-forward.md` §7.2
本节要写：给出"任何一层都长这样"的骨架——`hidden → attn_norm → token 混合器 → 残差 → post_attention_norm → FFN → 残差`——把注意力与 FFN 之外的公共部分（zero-centered RMSNorm、SiLU 门控 FFN 的 gate/up/down 三投影）先钉死，后面实现时才不会重复设计。

### 2.2 每四层一次的全注意力层 | 参考文献 `doc/07-forward.md` §7.3
本节要写：讲清全注意力层相对"教科书"变了什么：`attn_q` 输出两倍宽（query ∥ gate 交错）、带 sigmoid 的注意力门控、per-head RMSNorm 规范化 Q/K、部分 RoPE。读者将记住三个数字：head_count、head_count_kv（GQA）、RopeDim（部分旋转）。

### 2.3 其余三层：Gated-DeltaNet 线性注意力 | 参考文献 `doc/07-forward.md` §7.4
本节要写：用直觉（"用递归状态替代全部历史，不用存下所有 K"）引出线性注意力，再落到 delta rule 递推的四件套：状态矩阵 S、衰减 decay、β 误差修正、k 与 q 的 L2 归一化。这段不为代码，只为了让读者在实现第 11 章前已经知道每一步在算什么。

### 2.4 张量命名法：GGUF 里的名字就是设计文档 | 参考文献 `doc/06-model.md` §6.1、`doc/07-forward.md` §7.4.1
本节要写：把 Qwen3.5 全部张量名（`attn_qkv`/`ssm_conv1d`/`ssm_beta`/`ssm_out`/`ffn_gate_inp`…）当字典列出，说明"哪一维是行、哪一维是列、谁是 hidden 谁是 expert"由名字+末尾维度决定——因为后面 Chapter 3 的读取器和 Chapter 8 的加载器都要靠这套名字来认张量。

## 3. 第一份代码：读取 GGUF 权重文件

**本阶段意义**：开发者真正写下的第一段代码是"读文件"，因为模型本体在 GGUF 里，没有它一切都悬空。本章从二进制布局讲到实现的每个函数，并以一个"历史陷阱"（type 42 歧义）示范兼容层的存在价值。

### 3.1 GGUF 在磁盘上长什么样 | 参考文献 `doc/03-gguf.md` §3.2
本节要写：画一份内存布局图（magic/version/tensor_count/kv_count/kv 条目/tensor 信息/数据段对齐），并解释两个关键设计：kv 元数据先行使读取器要先扫一遍才能定位数据段；`general.alignment` 决定数据段起点，一切偏移从这里算。落地代码在 `GgufValueType` 枚举与 `GgufFile.Parse`。

### 3.2 用一个安全的游标顺序扫描 | 参考文献 `src/Qwen35Sharp.Gguf/GgufFile.cs` — `GgufFile`、`Cursor`、`Parse`
本节要写：展示如何用一个带越界检查的字节游标 `Cursor`（`ReadU8/ReadU16/ReadU32/ReadU64/ReadF32/ReadF64/ReadString` + `Ensure`）把"解析"降维成"顺序读"，并讨论为什么一切解析在 dispose 路径上走到公共清理——解析失败时 `catch → Dispose()` 保证 mmf 不被泄漏。

### 3.3 内存映射：把权重留在磁盘上 | 参考文献 `src/Qwen35Sharp.Gguf/GgufFile.cs` — `GgufFile` 构造函数、`DataBase`、`doc/03-gguf.md` §3.3
本节要写：解释 `MemoryMappedFile` + `AcquirePointer` 的取舍——16GB 权重不需要真进内存，回报给读者一个 `DataBase` 指针与 `GetString/GetInt32/GetInt32Array/GetUint32/GetFloat32` 元数据 API，张量字节原地不动。这是全书内存策略的第一块基石（配合第 8 章的按需拷贝）。

### 3.4 宽容的元数据读取：一切都能强转 | 参考文献 `src/Qwen35Sharp.Gguf/GgufFile.cs` — `CoerceInt64`、`CoerceFloat`、`CoerceString`、`ReadValue`、`ReadScalar`
本节要写：点出 GGUF 元数据的现实——不同导出法器有的存 u32、有的存 u64——所以读取器提供一套 `Coerce*` 宽容强转（整数 checked 收窄、整数→浮点），并画清 `GetInt32Array` 的元素逐个强转路径与缺省值语义，让读者明白"读错类型不炸"是兼容性设计而非偷懒。

### 3.5 张量清单：名字、类型、形状、偏移 | 参考文献 `src/Qwen35Sharp.Gguf/GgufTensorInfo.cs`、`src/Qwen35Sharp.Gguf/GgmlTensorType.cs`
本节要写：解释 `GgufTensorInfo`（Type/Offset/Shape/NumElements）如何把"文件里的每个张量"变成一张可寻址的目录，并强调 GGUF 的维度约定——**dim 0 是最内层（行长度），与 ggml 一致**；这个约定后面 GEMM 与量化全都在用。同时列出按 llama.cpp 对齐的完整类型枚举。

### 3.6 类型规范化的历史陷阱：type 42 到底是 Q2_0 还是 STQ | 参考文献 `src/Qwen35Sharp.Gguf/GgufFile.cs` — `CanonicalTensorType`、`tests/Qwen35Sharp.Tests/GgufTypeTests.cs — LegacyStride16Type42IsNormalizedWithoutBreakingQ2`
本节要写：用一个真实的兼容性事故讲"格式演进"：早期 HyMT2 Stride16 模型把 STQ 存成了当时的 type 42，而 llama.cpp 重排枚举后 42 是标准 Q2_0 了——于是本项目用元数据特征（file_type==41 且名称含 "Stride16"）消歧，普通 Q2_0 永不被误读。本节示范"格式层要容忍历史，但绝不能误读当下"，并附上测试如何用手写二进制锁住这个行为。

## 4. 量化：为什么权重能塞进内存，以及怎么解码

**本阶段意义**：模型加载进来后，大头是量化权重。开发者必须先彻底搞懂"每个格子几个 bit、谁在每 256 个值里存一个尺度"，才能写对第 5、6 章的内核。本章按 K 系 super-block 的家族关系逐个讲解。

### 4.1 从"1 个 f32"到"256 个值一个块"的思维转换 | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `Qk` 静态类
本节要写：先用 Q4_K 为例讲明"super-block = 256 值共享 d/dmin + 8 个子块各自 scale/min"的分层缩放思想，然后展示 `Qk` 里全部尺寸常量（`Qk.SuperBlock`、`Q4KSize`、`Q8KSize`、…），这是后面所有内核按字节精确取数据的"地图"。顺带说明这些块结构体为什么统一 `[StructLayout(Pack=1)]`。

### 4.2 Q4_K：128 nibble + 12 字节 scales | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `BlockQ4K`、`src/Qwen35Sharp.Kernels/Q4K.cs` — `Q4K.GetScaleMin`、`Q4K.UnpackScales`、`Q4K.DequantizeRow`
本节要写：把 Q4_K 的内存排布（`D/Dmin` fp16、`Scales[12]` 每子块各 6-bit 的 scale 与 min、`Qs[128]` 每字节两个 4-bit nibble）讲透，并给出解码一行 `y[2i]=d1·(Qs&0xF)−m1` 的公式；再解释 `PackSimple` 为什么只是测试用具（真正的量化发生在模型发布端，本书不需要）。

### 4.3 Q8_K：全程通用的激活中间格式 | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `BlockQ8K`、`src/Qwen35Sharp.Kernels/Q8K.cs` — `Q8K.QuantizeRow`、`Q8K.ScaleBlock`
本节要写：说明为什么需要第二种格式——decode 时激活向量要被量化成 Q8_K 再与 Q4_K/Q6_K/Q2 权重做点积（"权重 4bit、激活 8bit"是 llama.cpp 的经典搭配），`Bsums[16]` 又是给哪类修正用的（各子块和，喂给 Q2/Q6/STQ 的均值修正）；重点剖析 `ScaleBlock` 的定标规则："先 absmax、再取第一个出现的带符号极值"、`iscale=−127/max`、`MathF.Round` 最近取整、全零行写 0。

### 4.4 Q5_K 与 Q6_K：在精度与体积之间的两个档位 | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `BlockQ5K`/`BlockQ6K`、`src/Qwen35Sharp.Kernels/Q5K.cs`、`src/Qwen35Sharp.Kernels/Q6K.cs — Q6K.DotAvx2`
本节要写：先讲 Q5_K 的"第 5 位存进 Qh"技巧与解码公式，再讲 Q6_K 的"Ql 低 4 位 + Qh 高 2 位 + 每 16 值一个 sbyte 尺度、零中心值展开减 32"；这章讲清楚两句话：Q5_K 没有 panel 因此 prefill 只能退化成 GEMV，Q6_K 的展开要在第 6 章配合 −32 修正再深入。

### 4.5 低比特家族：Q2_0C 与 STQ1_0 | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `BlockQ2_0C`/`BlockSTQ1_0`、`src/Qwen35Sharp.Kernels/Q2_0C.cs`、`src/Qwen35Sharp.Kernels/STQ1_0.cs`
本节要写：讲两个更激进的设计：Q2_0C 每字节 4 个 2-bit、值域 {-3,-1,1,3} 无零（512 值/块、一个 fp16 尺度）；STQ1_0 稀疏三元——每 256 值含 64 组、每组恰好一个零加三个等幅值，用 32 项码本编码 2-bit 模式、再叠加符号位。重点让读者理解 STQ 的"索引→码本→符号"三步查表，为第 5 章的 `vpshufb` 码本路由做铺垫。

### 4.6 IQ4_XS：非线性网格的争议与价值 | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `BlockIQ4XS`、`src/Qwen35Sharp.Kernels/IQ4XS.cs`
本节要写：介绍非线性 4-bit 网格 `kvalues_iq4nl`（不是线性 0..15 而是 {-127,-104,…,113}）与每子块 6-bit 尺度的拼装方法 `ls=(ScalesL[..]&0xF)|((ScalesH>>2·ib)&3)<<4`、`dl=D·(ls−32)`；说明它为什么是 MoE 专家的首选低比特格式（体积小、保留动态范围），以及查表不走展开的好处。

### 4.7 fp16 与软件 cvtph：没有 F16C 也要向量化 | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `HalfBits.ToSingle/FromSingle/Load8`
本节要写：解释 d/dmin 这类 fp16 字段如何在不假设 F16C 指令的前提下批量装入——`BitConverter.UInt16BitsToHalf` 的标量极简路径，加上手写 AVX2 软件 `cvtph_ps` 的 `HalfBits.Load8`（处理有穷数、符号零），说明"无指令集假设仍可向量化"这一取舍。

---

# 第三阶段 计算内核

## 5. 最小计算单元：量化点积与 GEMV

**本阶段意义**：开发者造完"能读权重的引擎"后的第一步计算，是"一行权重 × 一行激活"的点积，以及把它并排拼起来、逐行输出的 GEMV。这是 decode 每生成一个 token 都会无限反复踩的路径，性能与正确性都从这一章滚雪球。

### 5.1 decode 的第一性原理，为什么它离不开新鲜的行向量 | 参考文献 `doc/05-kernels.md` §5.1、`doc/04-quantization.md` §4.2
本节要写：先厘清自回归的本质——每步只新来一个 token，所以面对的是 `1 × nIn` 的输入，所有权重行都要跟这一行点积，这就是 GEMV；由此引出 decode 的黄金路径"激活先量化一次成 Q8_K，然后每行点积"，为 5.2 的共享量化做铺垫。

### 5.2 Q8_K 激活量化：一次量化，处处复用 | 参考文献 `src/Qwen35Sharp.Kernels/Q8K.cs` — `Q8K.QuantizeRow`、`src/Qwen35Sharp.Kernels/Q4K.cs — Q4K.DequantizeRow`
本节要写：完整讲一行 f32 向量量化成一个 `BlockQ8K` 的细节（absmax 定标、`ScaleBlock` 的带符号极值规则、clamp 与最近取整、全零行写 0），并说明"量化不在每个 GEMM 里做，而在模型层的 `DecodeGateUp` 里做一次、几个矩阵共用"——这是性能的关键，也让本章和第 9 章咬合。

### 5.3 Q4_K × Q8_K 的点积：从标量到 AVX2 | 参考文献 `src/Qwen35Sharp.Kernels/VecDotQ4K.cs` — `VecDotQ4K.Dot`、`DotAvx2`
本节要写：先给一版朴素标量点积（遍历 super-block，每个子块反量化后乘加 + min 修正），再演进到 AVX2 版：`vpmaddubsw` 做 8×8→16 的乘加、用 SSE 低半槽累积 accMin、最后合并 D/Dmin 修正。刻意强调这版内核"int16 中间累加安全"的证明——每 short ≤ 8·15·127——给后面所有整数内核立下正确的溢出审计习惯。

### 5.4 其余格式的点积：Q6_K、Q2_0C、IQ4_XS、STQ1_0、Q5_K | 参考文献 `src/Qwen35Sharp.Kernels/Q6K.cs — Q6K.Dot/DotScalar/DotAvx2`、`Q2_0C.cs — Q2_0C.Dot/DotAvx2/DotExpanded/DotScalarForTest`、`IQ4XS.cs — IQ4XS.Dot`、`STQ1_0.cs — STQ1_0.DotAvx2/DotChunk16`、`Q5K.cs — Q5K.Dot`
本节要写：逐个过一遍不同格式点积的实现诀窍：Q6 的掩码提升（m3/m12/m48/mC0 把 Qh 的 2-bit 拼成高 nibble）；Q2_0C 的"字节→dword→独立字节展开"避免标量解包、以及多个标量参考锚点（`DotScalarForTest`/`DotExpanded`）的存在意义；IQ4_XS 直接查 `kvalues_iq4nl` 免展开；STQ1_0 的 `DotChunk16` + `CodebookLo/CodebookHi` 两张码本向量 + `SignMaskLut` 符号掩码查表，并在点积后减 Q8 激活和还原 −1/0/+1 语义。最后承认 Q5_K 只做往里叠加的普通标量+AVX2 版本。

### 5.5 硬件快车道：FastExp 与 SiLU | 参考文献 `src/Qwen35Sharp.Kernels/FastExp.cs` — `FastExp.ExpAvx2`、`FastExp.SiluAvx2`
本节要写：在本章插入一个"一路上每个角落都要用的基础件"——对 llama.cpp `ggml_v_expf/v_silu` 的精确移植：2^n 分解 + f32 多项式、max ~1.45 ulp、`|x|>88.38` 冲 +inf、`<−103.97` 冲 0；说明 softmax、SiLU、门控、FFN 四处共用它，并把"为什么宁可要快也不全精"的性能观讲给读者。

### 5.6 把它拼成 GEMV：逐行循环考量的缓存与并行 | 参考文献 `src/Qwen35Sharp.Kernels/MulMatQ4K.cs — MulMatQ4K.Gemv`、`src/Qwen35Sharp.Kernels/Q6K.cs — Q6K.Gemv`、`Q5K.cs — Q5K.Gemv`、`IQ4XS.cs — IQ4XS.Gemv`
本节要写：把第 5.3–5.4 的点积包上循环变成 GEMV：`nOut` 行权重各点积一次、按 worker 等分派发线程；给出 Q4_K 行权重如何按 super-block 步进、scratch 里临时存 Q8 行的做法，以及无 AVX2 时回退 `DotScalar` 的标量路径。这一步完成了"decode 一次能跑了"的第一个闭环。

### 5.7 正确性锚点：为什么每个内核都配一个标量兄弟 | 参考文献 `src/Qwen35Sharp.Kernels/Q2_0C.cs — DotScalarForTest`、`Q6K.cs — DotScalar`、`IQ4XS.cs — DequantizeRow`、`tests/Qwen35Sharp.Tests/IQ4XSTests.cs`
本节要写：讲清项目独有的验证纪律——每个 SIMD 内核都伴生一个可读的标量参考实现，单测以"参考对拍 + 相对容差"断言一致性（如 `IQ4XSTests.Dot_MatchesScalarDequant`）。这在"没有 GPU 能对答案"的从零项目里，是唯一能把"算得对"变成可回归事实的方法。

## 6. 一次算一批：面板重打包与 panel GEMM

**本阶段意义**：prefill 要一次消化一整段提示词，逐 token GEMV 太慢。开发者在这里做的关键决策是：**把位手术从热循环里搬走**——加载时把权重重排成"每 8 个输出行一组"的 panel 并预解码尺度，热循环只做乘加。这是全 book 性能上最重要的一次重构。

### 6.1 prefill 的痛点：为什么 GEMV 撑不起长提示词 | 参考文献 `doc/02-architecture.md` §2.4-D2、`doc/05-kernels.md` §5.1
本节要写：用吞吐视角说明同一个矩阵在 prefill 里要跟几十上百个 token 的激活相乘，逐行 GEMV 的重复解码与缓存抖动是主要浪费；由此引出以 4×8 为单位的张量化方案——一次处理 4 个 token、8 个输出列——为后续 panel 设计立论。

### 6.2 重新排布世界的这一刻：panel 到底是什么 | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `BlockQ4Kx8`/`BlockQ6Kx8`/`BlockQ2x8`/`BlockSTQ1_0x8`、`src/Qwen35Sharp.Kernels/RepackQ4K.cs — MakeBlockX8/Rows/BuildMeta`、`RepackQ6K.cs — Expand/MakeBlockX8/Rows`、`RepackQ2.cs — Rows`、`RepackSTQ.cs — Rows`
本节要写：把四个重打包器各用一张小示意图讲清"输入哪种块、输出哪种更宽的块、interleave 到哪里、scale 如何重排"：Q4_K 的 8 列交织 nibble + 96 字节 scales 重排 + 448 字节预解码 meta sidecar；Q6_K 展开成 u8 0..63 + ScalePairs；Q2 与 STQ 各自 8 列的 32 值 tile 布局。最后强调这全部发生在加载期、只读、一次性，热循环因此"零位手术"。

### 6.3 Q8_Kx4：把激活也排成 4 行的面板 | 参考文献 `src/Qwen35Sharp.Kernels/BlockLayout.cs` — `BlockQ8Kx4`、`src/Qwen35Sharp.Kernels/QuantizeQ8Kx4.cs — Quantize4x8/Quantize4x8FromRows/Quantize4x8Avx2/Quantize4x8Scalar/Quantize4x8Silu`
本节要写：解说 Q8 激活面板的排布（`Qs[1024]` 每 32 字节是 4 行 × 8 值、`Bsums` 按 `((t>>2)<<4)+(t&3)` 的 FillBsums 顺序），以及 4 个 token 一次并行量化的两种实现（AVX2 与标量），并预告 `Quantize4x8Silu` 这份融合版会在第 9 章 FFN 登场。

### 6.4 4×8 GEMM：热循环只做乘加 | 参考文献 `src/Qwen35Sharp.Kernels/GemmQ4K.cs` / `GemmQ4K.Avx2.cs` — `GemmQ4K.Gemm8x8/GemmScalar/GemmAvx2/ColTileBytes`、`GemmQ6K.cs — GemmQ6K.Gemm8x8/GemmScalar/GemmAvx2`、`Q2Panel.cs — Q2Panel.Gemv/Gemm`、`STQPanel.cs — STQPanel.Gemv/Gemm`
本节要写：深入 `GemmQ4K.Avx2` 最热的一段：对每个激活行 `vpbroadcastq`（纯加载，无 shuffle 端口争用）对着预解码 meta 做 `vpmaddubsw`，每 8 个乘加只配 2 ands/6 loads 的无端口竞争结构；Q6_K 用扩展 u8 面板 + `ShiftLeftLogical(c,5)` 一次把 −32 修正折进整数累加器；结尾讲 `ColTileBytes=128KB` 的 L2 列分块为何让权重切片驻留 L2、q8 行流式扫过。

### 6.5 一石二鸟：共享量化与三连乘 | 参考文献 `src/Qwen35Sharp.Kernels/MulMatPanel.cs — MulMatPanel.QuantizeAndGemm/SiluQuantizeAndGemm/PanelWeight`、`MulMatQ2.cs — MulMatQ2.QuantizeAndGemm/SiluQuantizeAndGemm/ColTileBytes`、`MulMatSTQ.cs — MulMatSTQ.QuantizeAndGemm/SiluQuantizeAndGemm`
本节要写：展示调度层如何让"一个并行区段量化一份 Q8 面板、连续跑至多三个矩阵"——`MulMatPanel.QuantizeAndGemm` 同时喂 `ffn_gate`/`ffn_up`（或 `attn_q/k/v`），`PanelWeight` 这个 struct 把 Q4/Q6 面板与输出地址打包成可堆叠的参数；`GemmQ4K.GemmAvx2` 与量化之间用自旋 barrier 同步，完全对齐 llama.cpp 的 `mul_mat` 行为。

### 6.6 调度路由：tokens==1 走 GEMV，否则走 GEMM | 参考文献 `src/Qwen35Sharp.Kernels/MulMatQ4K.cs` — `MulMatQ4K.Gemv/GemvPrequant/Gemm/QuantizeAligned/QuantizeAndGemmPrequant/GemmPrequant/PackedGroups`、`MulMatQ2.cs`、`MulMatSTQ.cs`
本节要写：把"统一入口"讲清楚：`MulMatQ4K.Gemm` 怎么用 `PackedGroups(nOut)` 切面板、处理 token 数不是 4 的倍数（补零到 4、展平、回拷）与 `nOut%8` 的尾部 GEMV 补算；`GemvPrequant`/`GemmPrequant` 又如何接受调用方**已经量好的** Q8 行/面板（第 9 章 q/k/v 共用的通道）。这是内核层最后一块拼图。

## 7. 让多核为我们工作：线程池、拓扑与内存

**本阶段意义**：单线程 SIMD 只用了 CPU 的一小部分。开发者在 GEMM/GEMV 站稳后，立刻补上"并行开关"与"内存地基"，因为没有它们，第 9 章的 prefill 会慢到没有意义。本章解决"开多少线程、怎么派发、内存在哪里活"三件事。

### 7.1 不该用 Environment.ProcessorCount：CPU 拓扑发现 | 参考文献 `src/Qwen35Sharp.Kernels/CpuTopology.cs` — `CpuTopology` 静态属性、`ChooseWorkerCount`、`CountPCores`、`ParseCpuList`、`LooksLikeVirtualFirmware`
本节要写：解释为什么"逻辑核数"是个陷阱——lockstep GEMM 会被 SMT/E-core 拖垮——然后讲清跨平台发现物理核心的方法（Windows 走 `GetLogicalProcessorInformationEx` 的效率类 + group 感知，Linux 读 `/sys/devices/system/cpu/`），以及 `ChooseWorkerCount` 的启发式：默认取物理 P-core 数、VM 中减半避免每物理核多一个 worker、`AutoHint` 把理由解释给人看。

### 7.2 极轻派发：job-id 轮询的专用线程池 | 参考文献 `src/Qwen35Sharp.Kernels/CpuThreadPool.cs` — `CpuThreadPool`（构造）、`For`、`Barrier`、`Dispose`
本节要写：讲这套线程池为什么比 `Parallel.For` 更适合锁步计算：固定线程 + `_work` 数组按 worker 切分 + `Interlocked.Increment(_jobId)` 唤醒 + 主线程自旋等 `_remaining` 归零；再加入 `Barrier()` 这个 ggml 式自旋屏障，使"同一批 `For` 里先量化、后 GEMM"能精确同步；空闲 worker 自旋 8192 次后 `AutoResetEvent.WaitOne()` 休眠，避免顺序算子（RMS/SiLU）烧 P-core。

### 7.3 内存的地基：64B 对齐与竞技场 | 参考文献 `src/Qwen35Sharp.Kernels/NativeBuffer.cs`、`src/Qwen35Sharp.Kernels/ScratchArena.cs`、`doc/05-kernels.md` §5.5
本节要写：介绍两个基础件：`NativeBuffer`（`NativeMemory.AlignedAlloc(64)` + `Span` 安全视图 + 终结器兜底）作为一切权重与中间量的载体；`ScratchArena`（6 个惰性扩容的 A..F 槽、不回收复用、调用方约定每层各用各的槽）作为激活的内存池。读者会明白：对齐不是洁癖，是 AVX2 load/store 正确性与性能的前提。

### 7.4 结营检查：第一阶段的三份成果能跑通什么 | 参考文献 `README.md`"构建/运行"节、`doc/01-overview.md` §1.6
本节要写：一处非源码的收尾——清点现已拥有的三块积木（GGUF 读取器、量化点积/GEMV、panel GEMM、线程池），并用"假想的最小 main：读模型文件 → 反量化一行 → 与一段随机输入做一次 GEMV"说明本书到这里已经"能算"，只差把它组织成真正的模型——为第四阶段设下悬念。

---

# 第四阶段 拼装模型

## 8. 把文件变回模型：配置抽取与权重加载

**本阶段意义**：前两阶段造的是"通用计算器"；从本章起进入"这台机器专属于 Qwen3.5"。开发者要做三件事：把 GGUF 元数据变成一份超参表、把散落的张量按名字认领进统一结构、把缓存与内存按模型尺寸一次性备好。

### 8.1 一份超参表：ModelConfig 的诞生 | 参考文献 `src/Qwen35Sharp.Model/ModelConfig.cs` — `ModelConfig.FromGguf`、`IsRecurrent`
本节要写：逐一展示 `FromGguf` 如何把 GGUF 键（`embedding_length`、`attention.head_count`、`ssm.conv_kernel`…）映射成强类型字段，并为词汇残缺的键提供缺省值（如 `key_length`/`state_size` 缺省取 headDim）；再给出派生维度（`LinearConvDim/LinearKeyDim/LinearValueDim/FullQDim/FullKvDim`）的推导过程，让读者看到"值从哪来"即"维度从哪算起"。

### 8.2 看张量决定层类型：比公式更可靠的判定 | 参考文献 `src/Qwen35Sharp.Model/ModelConfig.cs` — `ModelConfig.FromGguf`、`doc/06-model.md` §6.1"层类型判定（关键）"
本节要写：讲清本项目甩开 `full_attention_interval` 公式的洞察——**别猜，看权重在不在**：有 `attn_qkv.weight` 就是线性层、有 `attn_q.weight` 就是全注意力、都没有才回退公式。这对"新 checkpoint 不严格遵守 4 层间隔"的现实极具韧劲，也是配置层一次漂亮的容错设计。

### 8.3 认领张量：union 式 Weight 与加载流程 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.cs` — `Weight`、`Qwen35Model` 构造函数、`doc/06-model.md` §6.2
本节要写：图解 `Weight` 这个"union 式"结构体——同内存按 `GgmlTensorType` 解释为 Q4/Q5/Q6/Q2/STQ/IQ4XS/F32 或各自的 panel/meta——然后走一遍 `LoadWeights`：按形状算块数与字节、整块拷进 `NativeBuffer`、对 `nOut>=8 && nExpert==1` 的张量追加构建 panel、对 3D 专家张量记录 `NExpert`/`ExpertStride` 而不重打包；最后解释 `_buffers` 的所有权与 `Dispose` 统一释放为何不会泄漏。

### 8.4 缓存与状态：出生时就把内存备好 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.cs` — `EnsureCache`/`EnsureScratch`、`doc/06-model.md` §6.3
本节要写：把"全注意力的 KV 缓存（只给非循环层分配、`[pos][kvStride]`、倍增扩容、封顶 ContextLength）"与"线性层的固定状态（`_linState` delta-rule 矩阵、`_linConv` 滚动卷积窗口，与序列长度无关）"一起讲清；再讲 `EnsureScratch(seq, kvLen)` 的"最坏单层占用估算 + 4MB 余量 + 64B 对齐 Bump + 层间回卷"，说明如何在避免为上下文窗口预分配数 GB 的同时，保证前向途中指针稳定。

## 9. 前向骨架：一次 Forward 的发动机

**本阶段意义**：所有的"读文件"与"算内核"在本章第一次被缝合成 `Forward`。开发者从这里得到的是一条可跑的极简闭环（还没精心优化每个算子），它确立了"prefill 一次一段、decode 一次一个"的分野，以及"每层内部从 layerMark 回卷 Bump"的内存纪律。

### 9.1 Forward 主循环：把 token 变成 logits | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.cs` — `Forward`、`doc/07-forward.md` §7.1
本节要写：逐行走一遍 `Forward(int[] tokens)`：`_scratchUsed=0` → `EnsureCache/EnsureScratch` 按实际 size 整备 → `Embed` 填激活 `h` → 每层 `DecodeBlock`（seq==1）或 `PrefillBlock`（seq>1）→ `output_norm` 归一 → `Linear` 最后一行得到 logits → `_cacheLen += seq`。刻意强调"一次只更新缓存起点 start"与"层内重复 Bump、层间回卷"两条不变式，它们是后面所有内存复用成立的前提。

### 9.2 分派中转：Linear/DecodeLinear/GemvPrequant/F32Gemm | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.Decode.cs` — `Linear`/`DecodeLinear`/`GemvPrequant`/`Rms`/`RmsSerial`/`F32Gemm`/`Embed`
本节要写：把模型层的"翻译器"讲清楚：`Linear` 按 `w.Type` 的 `switch` 把调用交给对应内核；一套 `Decode*` 版本专供单 token；`GemvPrequant` 接收调用方已量好的 Q8 行（gate/up、q/k/v 复用的关键）；`F32Gemm` 用逐行 `Ops.DotF32` 兜住 router/共享门控等小张量；`Embed` 按 `token_embd.weight` 类型反量化并支持 `tokens>8` 并行。读者将看到"权重格式决定内核、CPU 能力决定 SIMD 宽度"的分层哲学。

### 9.3 逐点算子全家桶 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Ops.cs` — `RmsNormZeroCentered`/`GatedRmsNorm`/`Sigmoid`/`SoftplusScalar`/`Log1pSafe`、`src/Qwen35Sharp.Model/Ops.cs` — `RmsNorm`/`NeoXRoPE`/`DotF32`/`AxpyF32`/`SiLUMul`/`SiLUMulRange`/`AddInPlace`
本节要写：把前向里所有"非 GEMM"的逐点/规约算子集中讲解一次：zero-centered RMSNorm 的 `y=x·rsqrt(mean(x²)+eps)·(1+w)`（权重初始化为 0 时的关键差异）、三段的数值稳定 softplus、Llama 风格 vs Qwen 风格 norm、NeoXRoPE 的 cos/sin 表、以及 AVX2 版本的 dot/axpy/silu/add。这些是第 10、11、12 章会反复调用的基础件。

### 9.4 DecodeBlock 与 PrefillBlock：单层的统一画布 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.Decode.cs` — `DecodeBlock`、`src/Qwen35Sharp.Model/Qwen35Model.Prefill.cs` — `PrefillBlock`
本节要写：画出第 2 章"统一层结构"的代码化身：`n1=Rms(attn_norm)` → token mixer（线性 or 全注意力，各自两种 prefill/decode 变体）→ 残差 → `n2=Rms(post_attention_norm)` → FFN（MoE or dense gate/up/down）→ 残差。引出后三章将对这套画布做三次具体填充，也预告 FFN 的 `DecodeGateUp` 会用一次 Q8 量化喂两个 GEMV。

## 10. 全注意力层：标准 GQA 的落地

**本阶段意义**：这是模型里"最像教科书"的部分，也是第一块被填进画布的 token 混合器。开发者在实现它的过程中，学会处理交错投影、RoPE、per-head norm、GQA 这三件在 Qwen 系里"标配但不平凡"的细节。

### 10.1 投影与 q|gate 解交错 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.Prefill.cs` — `PrefillAttention`、`src/Qwen35Sharp.Model/Qwen35Model.Decode.cs` — `DecodeAttention`
本节要写：从 `attn_q.weight` 输出 2 倍宽讲起——GEMM 结果按头交错得 `[q_h0, gate_h0, q_h1, gate_h1,…]`，于是必须按头解交错成独立的 q 与 gate 缓冲，k/v 直接写进 KV 缓存 `start..start+seq`；给出 head-major `[seq][heads][dim]` 的布局约定，这是后续一切头相关计算的把手。

### 10.2 RoPE 与 per-head RMSNorm | 参考文献 `src/Qwen35Sharp.Model/Ops.cs` — `NeoXRoPE`、`doc/07-forward.md` §7.3.2
本节要写：解释 `Ops.NeoXRoPE` 如何按 `RopeDim` 做部分旋转（cos/sin 表 `freq_i=base^(−i/half)`、`EnsureRopeTable` 按需增长），以及为什么旋转后还要对每个头单独做一次 zero-centered RMSNorm（`attn_q_norm`/`attn_k_norm`）——Qwen3.5 的怪脾气之一，求解过程要在矩阵形态上做 `seq·heads` 行的 norm。

### 10.3 评分、softmax、加权：AttentionScores→SoftmaxCausal→AttentionCombine | 参考文献 `src/Qwen35Sharp.Model/Ops.cs` — `AttentionScores`/`SoftmaxCausal`/`AttentionCombine`、`src/Qwen35Sharp.Kernels/FastExp.cs — ExpAvx2`
本节要写：把注意力三段式逐个讲透：评分里 `Dot2x4`（2 个 q 行 × 4 个 k 行）的批处理为什么省 K 的重复加载；`SoftmaxCausal` 只归一可见前缀、尾部精确写 0、被减量 floor 到 −80 避免下溢污染归一；`AttentionCombine` 用 `Axpy64` 让 64 维输出驻留寄存器。同时给出 GQA：`group=heads/kvHeads`，每个 kv 头服务一组 q 头。

### 10.4 sigmoid 门控与输出投影 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Ops.cs` — `Sigmoid`、`Qwen35Model.Decode.cs — DecodeAttention`、`Prefill.cs — PrefillAttention`
本节要写：点出 Qwen3.5 相对 Qwen2/3 的改动——`out = attention_out · sigmoid(gate)` 逐元素门控，然后再 `attn_output` 投影；说明 prefill 走批量、decode 走单行的两条实现，以及这块为何是可以安全合并进 attention 输出的融合点。

### 10.5 KV 缓存的写入与截断：对话历史的账本 | 参考文献 `src/Qwen35Sharp.Model/KvCacheAlign.cs` — `KvCacheAlign.CommonPrefixLength/Plan`、`doc/07-forward.md` §7.5
本节要写：讲清 KV 缓存从 `start` 开始顺序写、`KvCacheAlign.Plan` 找公共前缀、全命中时重放最后 token 刷新 logits、分歧 tail 时截断到 common 的逻辑，也强调一个诚实的坑：**截断只回滚全注意力 KV，线性层状态不回滚**——所以当前实现约定"对话内缓存只增，/reset 才全清"。这是把注意力层真正接进多轮对话的一节。

## 11. Gated-DeltaNet：线性注意力的心脏

**本阶段意义**：这是全书唯一"普通推理引擎书里没有"的章节，也是本项目最重的数学投资。开发者要闭眼能把 delta rule 背出来：先衰减、读 kvMem、计算 β 修正、回写、再读输出——因为第 5 章的递给它的张量都是为这一步服务的。

### 11.1 从直觉到算式：为什么叫 delta-rule | 参考文献 `doc/07-forward.md` §7.4、`src/Qwen35Sharp.Model/GatedDeltaNetOps.cs — GatedDeltaNetOps` 头注释
本节要写：不写代码，只讲数学：状态 S 记忆"过去键值的加权和"；每个新 token 到达时，(a) 整块 S 乘以衰减 `decay=e^{g}`，(b) 用新键 k̂ 从 S 读出"记忆里已有什么" `kvMem=S·k̂`，(c) 算残差 `v−kvMem` 乘 β 作增量，(d) 按 k̂ 写回，(e) 用 q̂ 读出本轮输出。给读者一张"读-改-写"的时序图，比公式列表更有用。

### 11.2 投影张量与维度小宇宙 | 参考文献 `src/Qwen35Sharp.Model/ModelConfig.cs` — 派生维度、`doc/07-forward.md` §7.4.1
本节要写：把这层的九块投影（`attn_qkv`、`attn_gate`、`ssm_beta`、`ssm_alpha`、`ssm_a`、`ssm_dt.bias`、`ssm_conv1d.weight`、`ssm_norm.weight`、`ssm_out.weight`）的语义与形状一一对上，讲清 `LinearConvDim=2·keyDim+valueDim` 的拼接来源，以及 `ssm_a` 在 GGUF 里已烘烤成 `−exp(A)` 的陷阱（实现时不能再 exp 一次）。

### 11.3 因果深度卷积：卷积状态与滚动窗口 | 参考文献 `src/Qwen35Sharp.Model/GatedDeltaNetOps.cs` — `GatedDeltaNetOps.CausalConv1d`/`CausalConv1dUpdate`
本节要写：先讲 prefill 的整段卷积（channel-major 转置、左补 kernelSize−1、`out[t]=silu(Σ_j k[j]·in[t−j])`、回写最后 `stateLen` 个原始输入进窗口）；再讲 decode 的单步更新如何用滚动 `[channels][stateLen]` 窗口一次算完并推进窗口（index 0 最旧）。两种形态的对应关系是本章易错点，值得一张对照表。

### 11.4 β、α 与衰减：把门控变成数字 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Ops.cs` — `Sigmoid`/`SoftplusScalar`/`Log1pSafe`、`doc/07-forward.md` §7.4.4
本节要写：把 `beta = sigmoid(bOut)`、`g = ssm_a·softplus(α + ssm_dt.bias)` 的数值细节讲透——为什么 softplus 需要分三段、为什么 `GatedDeltaNetOps.RecurrentGatedDeltaRule` 里最终 `decay=exp(g)` 恒 ≤ 0——让读者理解这里的每一项都在维护"记忆只淡不增"的物理直觉。

### 11.5 递推的代码化：RecurrentGatedDeltaRule | 参考文献 `src/Qwen35Sharp.Model/GatedDeltaNetOps.cs` — `RecurrentGatedDeltaRule`、`DeltaStep`、`tests/Qwen35Sharp.Tests/GatedDeltaNetOpsTests.cs — RecurrentGatedDeltaRule_MatchesReference`
本节要写：逐段解读 `RecurrentGatedDeltaRule` 与私有 `DeltaStep`：head 外层循环、token 内层循环、L2 归一化（Qwen3.5 恒真：`q̂=q·(1/√d/‖q‖)`、`k̂=k/‖k‖`）、以及 `DeltaStep` 的四步读改写；并列出一个诚实的性能注脚——此处每 token 分配临时 `float[]` 且三 pass 是标量，是加速迭代的已知欠债点。本节同时预告：`GatedDeltaNetOpsTests` 是全书最重要的测试，第 15 章会详述它如何用独立标量实现对拍锁定正确性。

### 11.6 门控归一化与输出 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Ops.cs` — `GatedRmsNorm`、`Qwen35Model.Prefill.cs — PrefillLinearAttn`、`Qwen35Model.Decode.cs — DecodeLinearAttn`
本节要写：收尾线性层的最后两步：`GatedRmsNorm(out, ssm_norm, z)` 用 `norm(out)·silu(z)` 门控，再乘 `ssm_out.weight`；并在此处完整走一遍 `PrefillLinearAttn`/`DecodeLinearAttn` 的调用链，确认卷积→拆分→头重复→递推→门控→输出整条链在两种模式下的镜像结构。

## 12. 稀疏 FFN：只算被选中的专家

**本阶段意义**：dense 模型的 FFN 第 9 章已经会算了（gate/up 共享 Q8 + SiluQuantizeAndGemm），本章补的是 `qwen35moe` 变体：262 个专家里每 token 只算 8 个。开发者的核心问题是"如何在 3D 张量里定位某个专家的切片、并把 top-k 选路写得既对又快"。

### 12.1 MoE 的数学与一次计算的全流程 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.Moe.cs`，`doc/08-moe.md` §8.1
本节要写：给出公式 `out=Σ_k w_k·Expert_k(silu(gate_k(x))·up_k(x))` + 共享专家项，画出"router 打分→softmax→top-k→重归一化→逐专家 FFN→加权求和"的一次完整旅程，强调**重归一化必须在 softmax 之后**（Σw=1 的物理要求）——这是最容易写错的一步。

### 12.2 3D 专家张量：一个张量装下全员 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.cs — Weight.ExpertStride`、`Qwen35Model.Moe.cs — ExpertBase`、`Qwen35Model.Decode.cs — LoadWeights 相关段`
本节要写：讲为什么 `ffn_gate_exps/up_exps/down_exps` 是 `[nIn, nOut, experts]` 的 3D 张量、加载期为什么**不**重打包成 panel（decode 只激活 top-k，逐专家做 GEMV 更划算），以及 `ExpertBase(w,exp)=base+exp·ExpertStride` 的字节定位如何用 `ExpertStride=nOut·nb·blockBytes` 而非按 NOut 线性估计。

### 12.3 路由：TopKSoftmax 与共享专家门控 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.Moe.cs` — `TopKSoftmax`、`PrefillSparseMoe`/`DecodeSparseMoe`
本节要写：拆解 `TopKSoftmax` 的贪心实现（整行 softmax 后线性扫描选 nUsed 个、选中置 −1 排除、再重归一化），再讲共享专家的两样器：`ffn_gate_inp_shexp` 的标量 logit 与 `F32Gemm` 的实现，最后 `output += sigmoid(sharedGateLogit)·SharedFFN(x)`。

### 12.4 逐专家 GEMV 与性能取舍 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.Moe.cs` — `ExpertGateUp`/`ExpertDown`、`doc/08-moe.md` §8.5
本节要写：讲清 decode 时对选中的每个专家做两个 GEMV + SiLU 的形态，坦白承认当前"每次 GEMV 独立量化"与"逐专家串行"是已知热点，并画一张未来路线图——"同一专家被多个 token 选中时合并成 panel GEMM"是明确的优化方向。落点在于：MoE 的收益来自"只算 8/262"，工程上的修改空间也一直围绕这个分母。

---

# 第五阶段 用户可见的世界

## 13. 让文字进得去出得来：字节级 BPE 与聊天模板

**本阶段意义**：前四个阶段让"token 进、logits 出"成立；但用户接触的永远是最外围两层——文字↔token 的分词器，与"怎么组织输入消息"的聊天模板。这一章把引擎从"数学机器"变成"能受理中文对话的程序"。

### 13.1 预拆分特殊 token：别让 `<|assistant|>` 被 BPE 撕碎 | 参考文献 `src/Qwen35Sharp.Model/BpeTokenizer.cs` — `BpeTokenizer` 构造函数、`_specialIds`
本节要写：先用第三阶段前的凡人做法（直接对整段文本跑 BPE）演示"特殊 token 被合并破坏"的失败，再引入项目解法：构造时从 `tokenizer.ggml.token_type∈{3,4}` 与 bos/eos 收集 `_specialIds`，`Encode` 开头把它们从原文里预切出来（`fragments` 列表带预设 id），保证 `<|assistant|>` 这类 token 原样一个落一个，其余片段才进入 BPE。

### 13.2 预分词与字节编码：一统中文与英文 | 参考文献 `src/Qwen35Sharp.Model/BpeTokenizer.cs` — `ResolvePreTokenizer`、`ByteEncode`、`EncodeByte`、`AppendTokenBytes`
本节要写：讲三件事：预分词正则以词素为主（含 gpt2 的 `'s|'t|…| ?\p{L}+| ?\p{N}+|…` 变体）；字节编码如何把 UTF-8 字节映射到可见字符区间保证单 token 内可逆；以及 `Bpe` 的合并算法（从单字符开始反复找 rank 最低的相邻对、`bestIndex` 顺序拼词、无候选即停、缺席片断回退 `UnkId`）。中文/emoji/混合文本为何都能走同一管道，也在本节内获得解释。

### 13.3 Decode 与增量流式：把 id 还原成可显示的文字 | 参考文献 `src/Qwen35Sharp.Model/BpeTokenizer.cs` — `Decode`、`DecodeVisible`、`IsStop`
本节要写：讲清解码方向（`AppendTokenBytes` 反向查回字节、`Encoding.UTF8.GetString` 汇总，跳过越界 id），以及 `IsStop(id)=id==EosId||id<0` 如何构成停止判定；`DecodeVisible` 的存在是为了给 CLI 的第 14 章增量差分打印提供"已可见文本"的稳定基准，避免字节级乱码。

### 13.4 Qwen3.5 聊天模板：把消息拼成模型期待的格式 | 参考文献 `src/Qwen35Sharp.Model/ChatTemplate.cs` — `ChatTemplate.RenderQwen35`、`ChatMessage`
本节要写：逐条展示 `RenderQwen35` 对 system/user/assistant 不同角色的包装规则以及生成提示 `<|assistant|>\n thinking\n`（`enableThinking=false` 时空思考块的变体），并强调"assistant 内容不 Trim、user 要 Trim"这类细节为什么影响多轮一致。最后说明模板为什么是纯文本——模型本身不支持视觉占位。

## 14. 让它开口说话：CLI 与自回归生成循环

**本阶段意义**：一切计算最终汇入的根本循环是：`Forward([prefill…])` → 取 argmax → `Forward([t])` → 取 argmax → …。本章把这条循环、前缀缓存复用、流式输出与 REPL 体验以成品的形态交付，读者将看到"前面所有引擎零件"在用户面前第一次完整运转。

### 14.1 命令行入口与参数解析 | 参考文献 `src/Qwen35Sharp.Cli/Program.cs` — top-level 语句、`Args`（`Args.Get`/`Args.GetInt`）
本节要写：走读 `Program.cs` 的开场：`--model/--prompt/--system/--max-tokens/--threads` 五个参数、`Args.Get`/`Args.GetInt` 的默认值、启动横幅（路径、线程数与自动提示、AVX2/VNNI 能力、架构超参）如何打印给用户视觉反馈"引擎已就绪"。

### 14.2 一次回答的完整旅程：Reply | 参考文献 `src/Qwen35Sharp.Cli/Program.cs` — `Reply`、`ArgMax`
本节要写：跟随 `Reply` 逐句走：渲染模板 → `Tokenizer.Encode` → `model.AlignPrompt` 前缀复用 → `model.Forward(suffix)` 做 prefill → `ArgMax`→ 进入 `maxTokens` 次 decode 循环（`IsStop` 提前跳出）→ 收集并 `WriteDelta` 增量打印 → 把回复压回消息数组。结尾打印 prefill/decode 的 tok/s 统计，让第一个"能对话的完整程序"成为本章的交付物。

### 14.3 前缀缓存复用：多轮对话不再重复计算 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.cs` — `AlignPrompt`/`TruncateCache`/`ResetCache`、`src/Qwen35Sharp.Model/KvCacheAlign.cs`、`doc/09-tokenizer-cli.md` §9.3
本节要写：把第 10.5 的 `KvCacheAlign` 拿到 CLI 层看效果：每次新消息渲染出完整提示后如何与 `_cacheTokens` 对齐、只把 suffix 送进 `Forward`、全命中时重放最后一个 token 刷新 logits；同时诚实说明 `TruncateCache` 对线性层状态不回滚的设计取舍，所以用 `/reset` 全清来保证一致性。

### 14.4 流式输出、停止判定与 REPL | 参考文献 `src/Qwen35Sharp.Cli/Program.cs` — `WriteDelta`、`DecodeVisible`、`IsStop`、REPL 命令分支
本节要写：讲 `WriteDelta` 的增量差分小把戏（`next[visible.Length..]`）如何实现"边生成边打印且不吐半字节乱码"，再讲贪心 `ArgMax`（无采样）的现状与 `/reset`/`/exit`/`/help` 的 REPL 循环、以及 `Console.IsInputRedirected/IsOutputRedirected` 切批处理模式的兜底。读者在这里第一次作为用户用上面向对话的成品。

---

# 第六阶段 工程质量

## 15. 用测试锁住正确性：没有 GPU 也想对答案

**本阶段意义**：CPU 引擎没有第二份实现可以对照，正确性全靠自己证明。本章给出项目"参考对拍"的测试哲学与五组关键测试，其中 Gated-DeltaNet 对拍是全书的"镇书之宝"——它证明了整条线性注意力链从第一行到最后一行都是对的。

### 15.1 测试哲学：对拍，而非自证 | 参考文献 `tests/Qwen35Sharp.Tests/FastExpTests.cs`、`IQ4XSTests.cs`，`doc/10-testing.md` §10.2
本节要写：先把"测试哲学"立起来：每个数值测试都配一个独立、可读的标量/数学参考实现（"用 SUT 验证 SUT"是被明令禁止的），随机输入固定种子、带相对容差、做法先修正 fp16 尺度再喂给断言（`ForceValidScales` 防空 NaN/Inf），并且**刻意**依赖 AVX2+FMA 断言（旧 CPU 会跳过）以便数值断言真打在 SIMD 路径上。

### 15.2 镇书之宝：Gated-DeltaNet 递推对拍 | 参考文献 `tests/Qwen35Sharp.Tests/GatedDeltaNetOpsTests.cs` — `RecurrentGatedDeltaRule_MatchesReference`、`src/Qwen35Sharp.Model/GatedDeltaNetOps.cs — RecurrentGatedDeltaRule/DeltaStep`
本节要写：逐条解释这组测试如何构造随机 q/k/v/g/β、独立写一份**同样的数学但独立代码路径**的标量参考、逐元素比对误差 <1e-3、并同时校验状态 S 与输出 out；作者借它说明"为什么这是全项目最重要的测试"——任何未来重构（向量化、就地归一、chunk）都以它不破为前提。

### 15.3 数值内核与融合路径的守恒 | 参考文献 `tests/Qwen35Sharp.Tests/FastExpTests.cs`、`IQ4XSTests.cs`、`doc/10-testing.md` §10.1
本节要写：串讲三组数值测试：`FastExpTests.ExpAvx2_TracksMathF`（相对误差 <2e-5）与 `SiLUMul_TracksScalar`；融合 `Quantize4x8Silu_MatchesSiLUThenQuantize` 的逐字节守恒（融合版与"先 SiLU 再量化"必须毫无差别）；`IQ4XSTests.Dot_MatchesScalarDequant`（点积 vs 反量化后点积）。给读者一张"融合路径必须与朴素路径逐位一致"的验收表。

### 15.4 手写二进制文件：不依赖真权重的格式测试 | 参考文献 `tests/Qwen35Sharp.Tests/GgufTypeTests.cs` — `LegacyStride16Type42IsNormalizedWithoutBreakingQ2`、`BpeTokenizer` 相关
本节要写：展示如何用 `BinaryWriter` 手写最小 GGUF 测格式规范化（type 42 消歧两个分支各测一边、互不误读），以及 K 系/分词器测试为什么全部用合成数据（仓库不装 GB 级权重，CI 可跑）。顺带点出不是最大的坑：测试里 `ForceValidScales` 这类"随机数制造前先洗牌"的细节。

### 15.5 缓存对齐与线程划分的专项测试 | 参考文献 `tests/Qwen35Sharp.Tests/KvCacheAlignTests.cs`、`FastExpTests.cs — ThreadPool_ForSplitsWork`
本节要写：列出 `KvCacheAlignTests` 的五条用例（分歧 tail 保留前缀/缓存长于提示不丢前缀/全命中重放/空缓存整段/无重叠重置），以及 `ThreadPool_ForSplitsWork` 如何验证 `CpuThreadPool.For` 把 [0,32) 恰好覆盖且每项一次——一类容易被"能跑就行"漏掉的正确性。

### 15.6 覆盖盲区：诚实地承认没测什么 | 参考文献 `doc/10-testing.md` §10.4
本节要写：以一段"已知盲区"收尾：无端到端生成测试（需真权重）、MoE/STQ/Q2/Q5/Q6 的 panel 与 GEMV 尚无专项对拍、BPE 与聊天模板没有专项单测——并给出把盲区转成工作项的方法（用 `DotScalarForTest` 等参考锚点补齐）。这是工程书该有的自省。

## 16. 性能剖析：看见时间都去哪了

**本阶段意义**：能跑、且跑对之后，"慢"是自然浮现的下一座山。本章把项目的剖析手段、诊断开关与已知瓶颈原样整理，让读者学会"先定位再优化"，而不是凭感觉改进。

### 16.1 内建剖析计数器：算子级耗时的秒表 | 参考文献 `src/Qwen35Sharp.Model/Qwen35Model.cs` — `ProfileEnabled`/`TicksQ4`/`TicksQ2`/`TicksSTQ`/`TicksQ6`/`TicksAttnScore`/`TicksSoftmax`/`TicksAttnCombine`/`TicksRms`/`TicksRope`/`TicksSilu`/`TicksEmbed`/`TicksDelta`/`TicksConv`/`ResetProfile`
本节要写：讲清这套静态计数器如何用 `Stopwatch.GetTimestamp()` 包裹各算子、累加到静态 long、`ResetProfile` 清零、几轮后除以 `Frequency` 得到分布；同时诚实标明哪些计数器只在 decode 侧累计（`TicksRope/TicksDelta`），prefill 还有部分未插桩。

### 16.2 追踪与诊断：Q35_TRACE 与 Q35_DIAG | 参考文献 `doc/11-performance.md` §11.3、`doc/07-forward.md` §7.1、`doc/02-architecture.md` §2.4-D7
本节要写：展示 `Q35_TRACE=1` 打印每层每个子步骤进入/完成的 stderr 追踪，`Q35_DIAG=1` 每 3 层打印最后一 token 隐藏态 RMS 范数（归一化到 /hidden 的 |h|）——它是定位"第几层开始数值发散"的探针，并给出一个诊断实例的工作流。

### 16.3 性能设计的全景回顾 | 参考文献 `doc/11-performance.md` §11.1、`doc/02-architecture.md` §2.4、`doc/04-quantization.md` §4.3
本节要写：把全书散落的性能决策（加载期重打包、decode/prefill 分流、共享激活量化、L2 列分块、物理核心线程、标量回退兜底）收拢成"性能设计六板斧"，每一条回指其成形的章节，让读者形成自己的优化决策框架。

### 16.4 已知瓶颈与优化路线 | 参考文献 `doc/11-performance.md` §11.6、`doc/08-moe.md` §8.5
本节要写：如实列出后续的四个坑：MoE prefill 逐专家独立量化的批处理机会、Gated-DeltaNet 递推每 token 托管数组分配与三 pass 标量循环（就地归一、向量化 kvMem/read/write、分块复现）、prefill 部分路径未插桩、Q5_K/IQ4_XS/STQ1_0 退化为逐 token GEMV；并给每一位优化方向配一句"怎么做"的指引，把读者接回可动手的下一步。

---

# 第七阶段 边界与远方

## 17. 这台引擎的边界，以及它还能走向哪

**本阶段意义**：最后阶段是给读者一张"地图的空白处"：明确哪些能力不在本书射程内（视觉/投机解码/采样多样性/KV 量化），哪些是已知的工程取舍（AVX2 依赖、nIn 需 256 倍数、前缀缓存对线性层状态的局限），以及未来一旦想继续，该从哪些方向依次投入。

### 17.1 明确的非目标：哪些能力刻意不实现 | 参考文献 `doc/12-limitations.md` §12.1、`README.md`"尚未实现"
本节要写：用一张表格收拢全书没做的东西——视觉/多模态（无 3D 位置 id、无 M-RoPE 分段）、投机解码/MTP、分块 Gated-DeltaNet（顺序与分块数学等价，这是解释为什么"没做也不亏"的好机会）、CLI 采样（目前仅有贪心 argmax）、KV 量化（f32 省事但 decode 带宽高）——并对每一项给一句"如果要做，会动到哪一层"。

### 17.2 已知限制：能跑，但别在这些边上踩雷 | 参考文献 `doc/12-limitations.md` §12.2
本节要写：把工程上的六条已知约束一件件交代清楚：热路径依赖 AVX2+FMA（无 AArch64/NEON，Q2/STQ panel 在无 AVX2 时抛 `PlatformNotSupportedException`）、量化张量 nIn 须为 256 倍数、只有 `nOut>=8 && nExpert==1` 才重打包、STQ1_0 兼容标记的条件三条、前缀缓存截断不扰动线性层状态、`EnsureScratch` 启发式估算在极端组合下的 `InsufficientMemoryException`。全部讲"为什么是、怎么绕"。

### 17.3 未来路线：从这本书出发，能走向哪里 | 参考文献 `doc/12-limitations.md` §12.3、`doc/11-performance.md` §11.6
本节要写：按"低垂果实 → 工程中期 → 长远愿景"分出三层路线：MoE 批处理与 Gated-DeltaNet 就地归一（不动架构的优化）、分块复现与 KV 量化（动内存布局）、再到 ARM64/NEON 与完整采样器（把引擎推向更多人群），并回扣开头那句"从零自制"的初衷——这本书不是终点，是让读者有能力继续往里挖的起点。

---

## 附录计划

- **附录 A 内核速查表**：`BlockLayout.cs — Qk` 全部尺寸常量、各面板大小、以及 `GemmQ4K.ColTileBytes`/`MulMatQ2.ColTileBytes` 等调参值汇总成一张表，供读者在实现时照抄不踩坑。
- **附录 B 张量与元数据字典**：按 `ModelConfig.FromGguf` 与 `LoadWeights` 的实际读取路径，把每个 GGUF 键、每个张量名、默认值、类型要求汇成参照表，对应 `doc/03-gguf.md`/`doc/06-model.md`。
- **附录 C 测试索引**：把 `GatedDeltaNetOpsTests`、`FastExpTests`、`IQ4XSTests`、`GgufTypeTests`、`KvCacheAlignTests` 的每个用例与它守护的源码函数做双向索引，方便读者"改了代码→找哪个测试验证"。
- **附录 D 阅读路径地图**：一份从 `Program.cs` 出发、沿真实调用链（Reply→Forward→PrefillBlock/DecodeBlock→GemmQ4K→…）标注文件与行号的"代码徒步路线"，把全书几十个锚点收进一张可走的图。