# 4. 量化格式与数据布局（`BlockLayout.cs` + 各格式内核）

所有 K 系格式共享 **super-block = 256 个值**。`Qk` 静态类集中定义尺寸常量与
`Qk.SuperBlock`。块结构体全部 `[StructLayout(LayoutKind.Sequential, Pack = 1)]`
以便按字节精确对齐 GGUF 中相应块。

## 4.1 尺寸常量速查

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
| `Qk.STQ1_0x8Size` | `8·4 + 512` | 8 列 STQ panel |

## 4.2 源格式（GGUF 磁盘布局）

### Q4_K —— `BlockQ4K`（144 B/块）

```
D (u16 fp16)  Dmin (u16 fp16)  Scales[12]  Qs[128]
```

8 个子块 × 32 值；`Qs` 每字节两个 4-bit 值（低半字节在前）。d/dmin 是 fp16；
每子块的 `scale`/`min` 各 6-bit，用 llama.cpp `get_scale_min_k4` 的位打包
（`Q4K.GetScaleMin` / `Q4K.UnpackScales` 实现）。解码一行：

```
对每个子块 g：
  d1 = d*scale_g,  m1 = dmin*min_g
  y[2i]   = d1*(Qs[l] & 0xF) - m1
  y[2i+1] = d1*(Qs[l] >> 4)  - m1
```

`PackSimple` 是**仅供测试**的简单打包器（min-max 映射），不是 ggml 的重要性加权
量化器——GGUF 里的权重到达时已量化好，模型不需要真正的量化器。

### Q8_K —— `BlockQ8K`（292 B/块）

```
D (f32)  Qs[256] (sbyte)  Bsums[16] (short)
```

激活的标准中间格式。`Bsums[j]` = 第 j 组 16 个 `Qs` 之和，供 K 系点积做
min/dmin 修正与 Q2/Q6/STQ 的均值修正。`Q8K.QuantizeRow` 的定标规则：
先取 absmax，再取**第一个出现的带符号极值**（`ScaleBlock`，与 llama.cpp x86/repack.cpp
`q8_K_4x8` 一致），`iscale = -127/max`，clamp 到 ±127，并用 `MathF.Round` 最近取整。
全零行写 0 d 与清零。

### Q5_K —— `BlockQ5K`（176 B/块）

```
D (u16)  Dmin (u16)  Scales[12]  Qh[32]  Qs[128]
```

每值 5-bit：`Qh` 提供第 5 位（每字节 8 个头的掩码，逐子块左移）。解码公式：

```
q = (Qs[l] & 0xF) + ((Qh[l] & mask) ? 16 : 0)      // 低半字节
q = (Qs[l] >> 4) + ((Qh[l] & mask) ? 16 : 0)        // 高半字节
y = d*scale*(q) - dmin*min
```

Q5_K 没有 panel 重打包：prefill 退化为逐 token GEMV（`Q5K.Gemm` = for t { Gemv }），
输出偏小、实现简单。

### IQ4_XS —— `BlockIQ4XS`（136 B/块，8×32）

```
D (u16 fp16 网格尺度)  ScalesH (u16)  ScalesL[4]  Qs[128]
```

非线性 4-bit 网格 `kvalues_iq4nl = { -127,-104,-83,-65,-49,-35,-22,-10,
1,13,25,38,53,69,89,113 }`。每子块（32 值）一个 6-bit 尺度：

```
ls = (ScalesL[ib/2] >> 4*(ib%2)) & 0xF | ((ScalesH >> 2*ib) & 3) << 4
dl = D * (ls - 32)          // 有符号偏移
y  = dl * kvalues[qs 低/高 nibble]
```

`IQ4XS` 提供 `DequantizeRow`（embedding/调试）、`Dot`（decode 点积，直接查表
免去展开）、`Gemv`/`GemvPrequant`/`Gemm`（tokens 循环调 GEMV）。MoE 专家的首选
低比特格式。

### Q6_K —— `BlockQ6K`（210 B/块）

```
Ql[128]  Qh[64]  Scales[16] (sbyte)  D (u16 fp16)
```

每值 6-bit：`Ql` 低 4-bit + `Qh` 高 2-bit（两个 4-bit 通道拼 8-bit），每 16 值一个
`sbyte` 尺度（零中心，值 = 展开值 − 32）。AVX2 点积用
`m3/m12/m48/mC0` 掩码把 `Qh` 的四对 2-bit 提升成 4 位高 nibble
（`Q6K.DotAvx2`）。`DotAvx2x4` 把**一个 super-block 反量化一次**同时对 4 个 Q8 行
做点积，供 prefill 的 `GemmFromQ8Range` 4-token 步进使用。

### Q2_0C —— `BlockQ2_0C`（130 B/块，512 值）

```
D (u16 fp16)  Qs[128]   // 512 个 2-bit 码，每字节 4 个
```

`BlockLength = 512`。权重值映射 `((b>>shift)&3)*2 - 3` → **{-3,-1,1,3}**（无零），
无 min 修正。`Q2_0C.Dot` AVX2 版 `Dot256` 把 8 个打包字节展开成 8 个 dword 后再
各拆成独立字节（避免标量解包与临时权重存储）。decode 用 Q8_K 激活；`Q2_0C.DotScalarForTest`
与 `DotExpanded` 为参考实现。

### STQ1_0 —— `BlockSTQ1_0`（42 B/块，256 值）

```
Qs[32]  Sign[8]  D (u16 fp16)
```

**稀疏三元**：每 256 值含 64 个 stride-16 组，每组 4 个 lane，恰好一个零、三个等幅
值，由 32 项码本编码。码本每个条目是一个打包的 4-lane 2-bit 模式（0=−1,1=0,2=+1），
索引 = `(sign<<4) | slot`：

```
slot = (Qs[group>>1] >> ((group&1)*4)) & 0x0F
sign = (Sign[group>>3] >> (group&7)) & 1
qpack = Codebook[(sign<<4) | slot]
Lane(qpack, lane) = ((qpack >> (lane*2)) & 3) - 1
```

`STQ1_0.DotAvx2` 用 `vpshufb` 码本查找（`CodebookLo/CodebookHi` 两个向量分别用于
正/负符号），配合 256 项的 `SignMaskLut`（每字节的 0/0xFF 选择器）去掉栈分配与位
循环；0/1/2 编码点积后**减去 Q8 激活和**转成 −1/0/+1 语义
（`DotChunk16` + `activationSum`）。

## 4.3 Panel 重打包格式（供 prefill GEMM）

**动机**：让 AVX2 热循环只做乘加，不处理 nibble/sign/scales 位手术。以下 panel 都
在 `LoadWeights` 时构建一次。

### Q4_K panel —— `BlockQ4Kx8`（1152 B）+ meta `BlockQ4Kx8Meta`（448 B）

8 个 Q4_K 块（= 8 输出列 × 256 K）重排成一个 panel：

```
D[8] u16  Dmin[8] u16  Scales[96]  Qs[1024]
```

`Qs` 布局为列交织：`chunk j`（64 B）含列 0..7 × K 值 4j..4j+3，让一次
`vpbroadcastq` 激活广播喂 4 列。scales 96 字节是 8 列 × 每列 12 字节的原位重排
（每对子块一组）。

`BlockQ4Kx8Meta` 把每 panel 块的 f32 `D/Dmin` 与**已解码的 int16**
scales/mins 提前算好，存储顺序匹配 `vphaddw` 输出
`[c0 c0 c1 c1 c4 c4 c5 c5 | c2 c2 c3 c3 c6 c6 c7 c7]`，mins 交错
`[m0lo m0hi m1lo m1hi …]`。热循环 `GemmQ4K.Avx2` 因此零位手术。

### Q8_K 激活 panel —— `BlockQ8Kx4`（1168 B）

4 个 Q8_K 块（= 4 个 token）交织：

```
D[4] f32  Qs[1024]  Bsums[64] (short)
```

`Qs` 每 32 字节是 4 行 × 8 值；`Bsums` 重排成
`((t>>2)<<4) + (t&3)` 顺序（`FillBsums`），匹配 GEMM 的读取模式。
`QuantizeQ8Kx4.Quantize4x8` 一次并行量化 4 个 token（AVX2 与标量版，
`Quantize4x8FromRows` 是参考实现）。

### Q6_K panel —— `BlockQ6Kx8`（2848 B）

```
D[8] f32  Scales[16*16]  ScalePairs[8*16]  Qs[2048]
```

值**展开成 u8（0..63）**，`Qs` chunk j（32 B）= 列 0..7 × K 值 4j..4j+3，一次
`vpbroadcastd` 喂 8 列。`Scales[i]` = 第 i 个 16 值子块 8 列各重复两次；
`ScalePairs[p]` 用于 −32·Σ(sc_i·bsum_i) 修正的成对排列。

### Q2_0C panel —— `BlockQ2x8`（1056 B）

```
D[8] f32  Qs[1024]
```

8 列 Q2 值，仍 2-bit/权。每个 64B chunk 含列 0..7 × 8 字节；四个 2-bit plane 各
含连续 8 个 K 值组。一次激活广播喂 4 列。`Q2Panel.Gemv/Gemm` 消费它。

### STQ panel —— `BlockSTQ1_0x8`（544 B）

```
D[8] f32  Qs[512]
```

把码本槽解码成 0/1/2 的 2-bit 值，与 Q2 panel 相同的 32 值 tile 布局。
`STQPanel.Gemv/Gemm` 消费它；`RepackSTQ.Rows` 在加载期完成解码，让热循环零
nibble/sign 解包。STQ 每个 256 值块有**独立** fp16 尺度（不同于 Q2 共享 512 值尺度）。

## 4.4 重打包器（`Repack*`）

| 重打包器 | 输入 → 输出 | 说明 |
|----------|------------|------|
| `RepackQ2.Rows` | `BlockQ2_0C` → `BlockQ2x8` | 拆 2-bit 码、跨列/跨 plane 重排 |
| `RepackQ4K.MakeBlockX8` / `Rows` | `BlockQ4K` → `BlockQ4Kx8` | 8 列交织 nibble + scales 重排；`BuildMeta` 生成 meta sidecar |
| `RepackQ6K.Expand` / `MakeBlockX8` / `Rows` | `BlockQ6K` → `BlockQ6Kx8` | 展开为 u8 0..63，scales/pairs 重排 |
| `RepackSTQ.Rows` | `BlockSTQ1_0` → `BlockSTQ1_0x8` | 码本解码 → 0/1/2 2-bit panel |

所有重打包都是**行序 → panel 序**的纯拷贝重排，加载时一次性执行，占模型加载时间
的一小部分。

## 4.5 fp16 工具（`HalfBits`）

`ToSingle/FromSingle` 用 `BitConverter` 在 fp16 与 f32 间互转。`Load8` 是
**AVX2 软件 `cvtph_ps`**（对有穷正规数与符号零），用于 GGUF 的 d/dmin 批量装入，
在无 `F16C` 指令集假设下仍向量化。