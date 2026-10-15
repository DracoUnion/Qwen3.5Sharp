# 9. 分词器、聊天模板与 CLI

## 9.1 字节级 BPE 分词器（`BpeTokenizer.cs`）

完全由 GGUF 的 `tokenizer.ggml.*` 元数据驱动，不依赖外部库。

### 元数据读取

| 键 | 用途 |
|----|------|
| `tokenizer.ggml.tokens` | token 表 `_idToToken` |
| `tokenizer.ggml.merges` | 合并表 `_merges`（"a b" → rank） |
| `tokenizer.ggml.bos_token_id` / `eos_token_id` / `unknown_token_id` | 特殊 id |
| `tokenizer.ggml.token_type` | 3/4 型特殊 token 标记 |
| `tokenizer.ggml.pre` | 预分词器名（默认 `gpt2`） |

`_specialIds` = token_type ∈ {3,4} 或等于 bos/eos id 的集合。

### Encode

1. **特殊 token 预拆分**：把每个已知特殊 token 的字符串从输入文本中先切出来，
   其余片段继续做 BPE。避免特殊 token（如 `<|assistant|>`、`<think>`）被合并破坏。
2. **预分词**：默认（gpt2/Qwen 风格）正则：
   `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`。
3. **字节编码**：`ByteEncode` 把 UTF-8 字节映射到 GPT-2 的可见字符区间
   （0x20 以下 +0x100、0x7F–0xA0 +0xA2、0xAD→U+0143 等），保证单 token 内字节
   可逆。
4. **BPE 合并**：从单字符开始，每次找 `_merges` 中 rank 最低的相邻对合并；
   无候选即停。找不到的片断回退 `UnkId`。

### Decode / DecodeVisible

把 id 逐 token 反向字节映射回原始字节，最后 UTF-8 解码。`DecodeVisible` 即
`Decode`。CLI 用它做**增量流式输出**（见下）。

### 停止判定

`IsStop(id) = id == EosId || id < 0`。Qwen3.5 的生成以 `eos_token_id`（`<|endoftext|>`
/ 思考结束 `</s>` 对应 id）作为停止条件。

## 9.2 聊天模板（`ChatTemplate.cs`）

`RenderQwen35(messages, addGenerationPrompt = true, enableThinking = true)`：

* system 消息 → `<|system|>\n…\n<|/system|>\n`。
* user → `<|user|>\n…\n<|/user|>\n`（`Trim()`）。
* assistant → `<|assistant|>\n…<|/assistant|>\n`（不 Trim，保留已生成内容原样）。
* 生成提示：`<|assistant|>\n<think>\n`；`enableThinking=false` 时改为
  `<think>\n\n</think>\n\n`（空思考块），用于关闭思考。

模板是纯文本（无视觉/多模态占位）。CLI 每轮把历史 `messages` 重新渲染成提示词。

## 9.3 CLI（`Qwen35Sharp.Cli/Program.cs`）

Top-level statements 程序。

### 参数（`Args` 静态解析）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `D:\_\model\Qwen3.5-4B-Q4_K_M.gguf` | 模型路径 |
| `--prompt` | （空 → REPL） | 单轮提示词 |
| `--system` | （空） | 系统消息 |
| `--max-tokens` | 256 | 最多生成 token 数 |
| `--threads` | 0（自动） | 线程数 |

启动打印模型路径、线程数与自动提示、AVX2/VNNI 支持、架构超参。

### 生成循环（`Reply`）

```
messages 追加 user
rendered = ChatTemplate.RenderQwen35(messages)
promptIds = Tokenizer.Encode(rendered)
align = model.AlignPrompt(promptIds)          // 前缀复用
logits = model.Forward(align.Suffix)          // prefill
token = ArgMax(logits)
循环 maxTokens 次：
  若 IsStop(token) 跳出
  logits = model.Forward([token])             // decode
  token = ArgMax(logits)
  追加并 WriteDelta（流式输出）
messages 追加 assistant 回复
打印 prefill/decode 计时与 tok/s
```

* **流式输出**：`WriteDelta` 用 `DecodeVisible(generated)` 的**增量**差分打印
  （`next[visible.Length..]`），避免字节级乱码、实现流式效果。
* **贪心**：`ArgMax` 直接取最大 logits（无采样）。
* **REPL**：`/reset`（清消息 + `ResetCache`）、`/exit`、`/quit`、`/help`。
  `Console.IsInputRedirected` / `IsOutputRedirected` 时切换批处理模式（不打印
  提示符、一次性输出完整文本）。

### 输出统计

```
prefill tokens=N (cached=C)  X ms  Y tok/s  |  decode tokens=M  X ms  Y tok/s
```

`prefill tok/s` 只算未缓存部分（`align.Suffix.Length`）。