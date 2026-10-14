# Qwen35Sharp

C# / .NET 10 CPU inference engine for **Qwen3.5** GGUF models, built on the
same skeleton as [HyMT2Sharp](../HyMT2Sharp): a memory-mapped GGUF reader plus
AVX2/AVX-VNNI quantized GEMM/GEMV kernels.

Qwen3.5 is a *hybrid* transformer: every fourth layer (1-indexed, i.e. layers
3, 7, 11, …) is a standard GQA attention layer with a sigmoid attention gate,
and the remaining layers are **Gated-DeltaNet** linear-attention layers
(a.k.a. gated SSM). This project implements both, plus the interleaved
SiLU-FFN, from a `.gguf` file.

## Layout

| Project | Purpose |
|---------|---------|
| `Qwen35Sharp.Gguf` | Memory-mapped GGUF v2/v3 reader (tensor + metadata). |
| `Qwen35Sharp.Kernels` | Quantized `x @ Wᵀ` GEMM/GEMV kernels (Q4_K, Q2_0C, STQ1_0, Q5_K, Q6_K), SIMD pointwise helpers, `CpuThreadPool`. |
| `Qwen35Sharp.Model` | The Qwen3.5 model: weight loading, KV + recurrent-state caches, prefill & decode forward, BPE tokenizer, chat template. |
| `Qwen35Sharp.Cli` | Interactive CLI that tokenizes a prompt, prefill + decode loop, and streams the answer. |
| `Qwen35Sharp.Tests` | xunit tests, including a reference-validated Gated-DeltaNet recurrence. |

## Building

Requires the .NET 10 SDK (the project targets `net10.0`).

```
dotnet build Qwen35Sharp.slnx
dotnet test  tests/Qwen35Sharp.Tests/Qwen35Sharp.Tests.csproj
```

On machines where the .NET 10 SDK is installed to the user profile rather than
`C:\Program Files`, point MSBuild at it:

```
$env:DOTNET_ROOT = "$env:USERPROFILE\.dotnet"
dotnet build Qwen35Sharp.slnx
```

## Running

```
dotnet run --project src/Qwen35Sharp.Cli -- ^
  --model "D:\_\model\Qwen3.5-4B-Q4_K_M.gguf" ^
  --prompt "Say hello"
```

An interactive REPL opens when `--prompt` is omitted; `/reset` clears the
conversation, `/exit` quits.

## Qwen3.5 architecture notes

Read from `general.architecture = "qwen35"`:

* `qwen35.embedding_length`        → hidden size
* `qwen35.attention.head_count`   → full-attn heads
* `qwen35.attention.head_count_kv`→ GQA kv heads
* `qwen35.block_count`            → number of layers
* `qwen35.full_attention_interval`→ every Nth layer is full attention
  (default 4); the others are Gated-DeltaNet.
* `qwen35.ssm.conv_kernel` / `inner_size` / `state_size` /
  `time_step_rank` / `group_count` → Gated-DeltaNet dimensions.
* `qwen35.rope.*`                 → RoPE base / dimension count.
* `qwen35.attention.layer_norm_rms_epsilon` → RMSNorm ε.

Per layer:

* **All layers**: `blk.L.attn_norm` → token mixer → residual →
  `blk.L.post_attention_norm` → SiLU MLP → residual.
* **Full-attention layer**: `attn_q` (2× output: query ∥ gate), `attn_k`,
  `attn_v`, `attn_q_norm`, `attn_k_norm` (per-head RMSNorm), partial
  RoPE, GQA scaled-softmax, then `out = out * sigmoid(gate)` and `attn_output`.
* **Gated-DeltaNet layer**: `attn_qkv`, `attn_gate`, `ssm_conv1d` (causal
  depthwise conv1d + SiLU), `ssm_beta` (= β), `ssm_alpha` (= α, with
  `ssm_dt` bias → softplus), `ssm_a` (= log A, → −exp(A)), gated-RMSNorm
  (`ssm_norm`), `ssm_out`. The delta-rule recurrence runs over the sequence
  (prefill) or a single token (decode), updating a per-layer
  `[heads][k_dim][v_dim]` state buffer; the conv state is a rolling
  `[k-1][conv_dim]` window.

`output.weight` (or tied `token_embd.weight`) produces the final logits.

## What is *not* implemented

* **Vision / multimodal** (`Qwen3_5VisionModel`, 3D position ids, M-RoPE
  sections) — this is a text-only engine. The 1-D `RopeDim` in
  `ModelConfig` is the partial-rotary dimension; full M-RoPE section
  handling is out of scope.
* **MoE** FFN — Qwen3.5 dense uses a standard SiLU FFN.
* **Speculative decoding / MTP**.
* The chunked Gated-DeltaNet path — only the sequential recurrence is
  used (equivalent, matches llama.cpp's per-token decode and is correct for
  arbitrary-length prefill).

## License

Apache-2.0.
