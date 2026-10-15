# 3. GGUF 读取器（`Qwen35Sharp.Gguf`）

## 3.1 职责

`Qwen35Sharp.Gguf` 是文件格式层，把 GGUF v2/v3 二进制文件的字节暴露成：

* 元数据键值查询（`GetString` / `GetInt32` / `GetInt32Array` / `GetUint32` /
  `GetFloat32` …），统一处理各标量类型之间的数值强转；
* 张量表 `Tensors`：名字 → `GgufTensorInfo`（类型、偏移、形状、元素数）。

**不**做反量化、不做权重预处理。张量数据留在内存映射中，通过
`DataBase + Offset` 直接寻址。重量级的拷贝与重打包在 `Qwen35Sharp.Model.LoadWeights`
完成。

## 3.2 文件格式（GGUF v2/v3）

GGUF 布局（与 llama.cpp `gguf.h` 一致）：

```
+----------------------+
| magic "GGUF"  (u32)   |
| version       (u32)   |
| tensor_count  (u64)   |
| kv_count      (u64)   |
| kv 条目 × kv_count    |    每个 kv：key(string) + 值（u32 type + payload）
| alignment 读取 key    |    general.alignment（默认 32，必须为 2 的幂）
| tensor 信息 × tensor_count |
+----------------------+
| pad 到 alignment     |
| 数据段（大端对齐，张量偏移从此计算） |
+----------------------+
```

每个 tensor 信息条目：

```
name（string）· n_dims（u32）· shape[d]（i64 × n_dims，d=0 最内层=行长度）
type（i32）· offset（u64，相对数据段起点）
```

值类型（`GgufValueType`，私有枚举）覆盖 GGUF 全部 13 种：

| id | 类型 | id | 类型 |
|----|------|----|------|
| 0 | Uint8 | 7 | Bool |
| 1 | Int8 | 8 | String |
| 2 | Uint16 | 9 | Array |
| 3 | Int16 | 10 | Uint64 |
| 4 | Uint32 | 11 | Int64 |
| 5 | Int32 | 12 | Float64 |
| 6 | Float32 |    |      |

`Array` 条目带元素类型与长度，返回 `object[]`（禁止嵌套数组）。找散列表键用
`StringComparer.Ordinal`，加速并避免区域差异。

## 3.3 解析实现（`GgufFile`）

构造 `GgufFile(string path)`：

1. 打开只读 `FileStream`，最小 24 字节校验。
2. `MemoryMappedFile.CreateFromFile` + `CreateViewAccessor`，`AcquirePointer` 拿到
   文件基址 `_fileBase`。
3. `Parse(length)` 用一个只读 `Cursor`（进位指针 + 越界检查 `Ensure`）顺序扫描：
   magic/version/tensor_count/kv_count → kv → 计算 `general.alignment` → tensor 信息
   → 数据段起点 `DataBase = _fileBase + Align(cur.Offset, alignment)`。
   tensorCount/kvCount 超过 `int.MaxValue` 视为损坏；维度 1..4 限制；负维度报错。
4. 任何解析失败走 `catch → Dispose()` 清理 mmf 后重新 throw。

`Cursor` 用 `Unsafe.ReadUnaligned<T>` 读取小端标量（x86/ARM 上未对齐读安全），string
按 `u64 长度 + UTF-8` 读取，全程边界检查 `_pos + bytes > _length` 即
`EndOfStreamException`。

`Dispose`：`ReleasePointer` → `_view.Dispose()` → `_mmf.Dispose()`，幂等。

### 解码与访问

* `GetInt32Array` 把元素逐个 `CoerceInt64` 强转并 `checked` 收窄到 `int`。
* `GetUint32` 无默认值版本在缺失或非整数时抛 `KeyNotFoundException`；有默认值版本
  返回默认。
* `CoerceInt64` 接受全部整数类型并做 checked 收窄；`CoerceFloat` 接受 float/double
  且把整数也转成浮点。这种宽容强转让 GGUF 里不同产生方（有些存 u32，有些存 u64）
  的元数据都能读出来。

## 3.4 类型规范化（关键兼容性逻辑）

GGUF 的 `ggml_type` 枚举在 llama.cpp 演进中重排过。`GgmlTensorType` 收录当前
llama.cpp PR #22836 的规范值，其中：

* `Q2_0C = 40`（本项目 / HyMT2 的紧凑 2-bit 格式）
* `Q2_0  = 42`
* `STQ1_0 = 43`（三元稀疏块，Sherry）

**历史问题**：早期 HyMT2 Stride16 模型的原始 type 42 在 llama.cpp 重排前代表 STQ，
现在与标准 Q2_0 冲突。`CanonicalTensorType` 用元数据特征消歧：

```
rawType == 42 && general.file_type == 41
&& general.name 包含 "Stride16"
  → 规范化为 STQ1_0
否则 → (GgmlTensorType)rawType
```

这样保证普通 Q2_0 文件永远不会被误读成 STQ 字节。对应单测
`GgufTypeTests.LegacyStride16Type42IsNormalizedWithoutBreakingQ2`。

## 3.5 类型枚举（`GgmlTensorType`）

与 llama.cpp `ggml.h` 对齐的完整列表（含计算/中间类型）：

```
F32=0  F16=1  Q4_0=2  Q4_1=3  Q5_0=6  Q5_1=7  Q8_0=8  Q8_1=9
Q2_K=10  Q3_K=11  Q4_K=12  Q5_K=13  Q6_K=14  Q8_K=15
IQ2_XXS=16  IQ2_XS=17  IQ3_XXS=18  IQ1_S=19  IQ4_NL=20  IQ3_S=21
IQ2_S=22  IQ4_XS=23
I8=24  I16=25  I32=26  I64=27  F64=28  IQ1_M=29  BF16=30
TQ1_0=34  TQ2_0=35  MXFP4=39
Q2_0C=40  Q2_0=42  STQ1_0=43
```

读取器本身不限制支持的种类；能否被模型/内核消费由下游 dispatch 决定
（见 `06-model.md` / `05-kernels.md`）。

## 3.6 `GgufTensorInfo`

`readonly struct`：

* `Type`：规范后的 `GgmlTensorType`。
* `Offset`：相对 `GgufFile.DataBase` 的字节偏移。
* `Shape`：GGUF 维度数组，**dim 0 最内层（行长度）** 与 ggml 一致。
* `NumElements`：各维 checked 相乘。

重量在 `LoadWeights` 时据此分配 `NativeBuffer` 并整段拷贝（见 `06-model.md`）。