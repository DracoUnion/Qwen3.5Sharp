using System.Diagnostics;
using Flygon.Gguf;
using Flygon.Kernels;

namespace Flygon.Model;

public sealed unsafe partial class Qwen35Model
{
    // =========================================================================
    // Shared dispatch helpers
    // =========================================================================

    private void Linear(float* input, string name, float* output, int nIn, int nOut, int tokens, string? fallback = null)
    {
        if (!_weights.TryGetValue(name, out Weight w) && fallback != null)
            _weights.TryGetValue(fallback, out w);
        switch (w.Type)
        {
            case GgmlTensorType.Q4_K:
                MulMatQ4K.Gemm(w.Packed, w.Q4, input, output, nIn, nOut, tokens, _pool, _gemmScratch, w.Meta);
                break;
            case GgmlTensorType.Q2_0C:
                MulMatQ2.Gemm(w.Packed2, w.Q2, input, output, nIn, nOut, tokens, _pool, _gemmScratch);
                break;
            case GgmlTensorType.STQ1_0:
                MulMatSTQ.Gemm(w.PackedSTQ, w.STQ, input, output, nIn, nOut, tokens, _pool, _gemmScratch);
                break;
            case GgmlTensorType.Q5_K:
                Q5K.Gemm(w.Q5, input, output, nIn, nOut, tokens, _pool);
                break;
            case GgmlTensorType.Q6_K:
                Q6K.Gemm(w.Q6, input, output, nIn, nOut, tokens, _pool, _gemmScratch);
                break;
            default:
                throw new NotSupportedException($"{name} type {w.Type} cannot be used as a linear weight.");
        }
    }

    private void DecodeLinear(float* input, string name, float* output, int nIn, int nOut)
    {
        Weight w = _weights[name];
        switch (w.Type)
        {
            case GgmlTensorType.Q4_K:
                MulMatQ4K.Gemv(w.Q4, input, output, nIn, nOut, _pool, _gemmScratch);
                break;
            case GgmlTensorType.Q2_0C:
                MulMatQ2.Gemv(w.Packed2, w.Q2, input, output, nIn, nOut, _pool, _gemmScratch);
                break;
            case GgmlTensorType.STQ1_0:
                MulMatSTQ.Gemv(w.PackedSTQ, w.STQ, input, output, nIn, nOut, _pool, _gemmScratch);
                break;
            case GgmlTensorType.Q6_K:
                Q6K.Gemv(w.Q6, input, output, nIn, nOut, _pool, _gemmScratch);
                break;
            default:
                throw new NotSupportedException($"{name} type {w.Type} is not a decode GEMV weight.");
        }
    }

    private void GemvPrequant(Weight w, BlockQ8K* x, float* y, int nIn, int nOut)
    {
        switch (w.Type)
        {
            case GgmlTensorType.Q2_0C:
                MulMatQ2.GemvPrequant(w.Packed2, w.Q2, x, y, nIn, nOut, _pool);
                break;
            case GgmlTensorType.STQ1_0:
                MulMatSTQ.GemvPrequant(w.PackedSTQ, w.STQ, x, y, nIn, nOut, _pool);
                break;
            default:
                MulMatQ4K.GemvPrequant(w.Q4, x, y, nIn, nOut, _pool);
                break;
        }
    }

    private void Rms(string name, float* x, float* y, int rows, int dim)
    {
        Weight w = _weights[name];
        Ops.RmsNorm(x, w.F32, y, rows, dim, Config.Eps, _pool);
    }

    private void RmsSerial(string name, float* x, float* y, int rows, int dim)
    {
        Weight w = _weights[name];
        Ops.RmsNorm(x, w.F32, y, rows, dim, Config.Eps, pool: null);
    }

    private void Embed(int[] tokens, float* dest)
    {
        Weight w = _weights["token_embd.weight"];
        int hidden = Config.HiddenSize;
        if (hidden % Qk.SuperBlock != 0)
            throw new NotSupportedException("token embedding inner dim must be a multiple of 256.");
        int nb = hidden / Qk.SuperBlock;
        int n = tokens.Length;
        void Row(int t)
        {
            float* row = dest + t * hidden;
            switch (w.Type)
            {
                case GgmlTensorType.Q4_K:
                    Q4K.DequantizeRow(w.Q4 + tokens[t] * nb, row, hidden);
                    break;
                case GgmlTensorType.Q5_K:
                    Q5K.DequantizeRow(w.Q5 + tokens[t] * nb, row, hidden);
                    break;
                case GgmlTensorType.Q6_K:
                    Q6K.DequantizeRow(w.Q6 + tokens[t] * nb, row, hidden);
                    break;
                case GgmlTensorType.Q2_0C:
                    Q2_0C.DequantizeRow(w.Q2 + tokens[t] * nb, row, hidden);
                    break;
                case GgmlTensorType.STQ1_0:
                    STQ1_0.DequantizeRow(w.STQ + tokens[t] * (hidden / STQ1_0.BlockLength), row, hidden);
                    break;
                case GgmlTensorType.F32:
                    Buffer.MemoryCopy(w.F32 + tokens[t] * hidden, row, hidden * sizeof(float), hidden * sizeof(float));
                    break;
                default:
                    throw new NotSupportedException($"token_embd type {w.Type}");
            }
        }

        if (n <= 8)
        {
            for (int t = 0; t < n; t++)
                Row(t);
        }
        else
        {
            _pool.For(n, (int worker, int workers) =>
            {
                int begin = n * worker / workers;
                int end = n * (worker + 1) / workers;
                for (int t = begin; t < end; t++)
                    Row(t);
            });
        }
    }

    private float* WeightF32(string name)
    {
        if (!_weights.TryGetValue(name, out Weight w))
            throw new KeyNotFoundException(name);
        if (w.Type != GgmlTensorType.F32)
            throw new NotSupportedException($"{name} is not an F32 weight");
        return w.F32;
    }

    private float* ConvKernel(int layer)
    {
        if (!_weights.TryGetValue($"blk.{layer}.ssm_conv1d.weight", out Weight w))
            throw new KeyNotFoundException($"blk.{layer}.ssm_conv1d.weight");
        if (w.Type != GgmlTensorType.F32)
            throw new NotSupportedException("Qwen3.5 ssm_conv1d.weight must be F32");
        return w.F32;
    }

    // =========================================================================
    // Decode (1 token)
    // =========================================================================

    private void DecodeBlock(float* hidden, int layer, int start)
    {
        const int seq = 1;
        int hiddenSize = Config.HiddenSize;
        float* n1 = (float*)Bump((nuint)hiddenSize * sizeof(float));
        float* attn = (float*)Bump((nuint)hiddenSize * sizeof(float));
        RmsSerial($"blk.{layer}.attn_norm.weight", hidden, n1, seq, hiddenSize);

        if (Config.IsRecurrent[layer])
            DecodeLinearAttn(n1, attn, layer, start);
        else
            DecodeAttention(n1, attn, layer, start);
        Ops.AddInPlace(hidden, attn, hiddenSize, pool: null);

        float* n2 = (float*)Bump((nuint)hiddenSize * sizeof(float));
        RmsSerial($"blk.{layer}.post_attention_norm.weight", hidden, n2, seq, hiddenSize);
        float* gate = (float*)Bump((nuint)((long)Config.FfnSize * sizeof(float)));
        float* up = (float*)Bump((nuint)((long)Config.FfnSize * sizeof(float)));
        float* down = (float*)Bump((nuint)hiddenSize * sizeof(float));
        DecodeGateUp(n2, gate, up, layer);
        Ops.SiLUMul(gate, up, Config.FfnSize, pool: null);
        DecodeLinear(gate, $"blk.{layer}.ffn_down.weight", down, Config.FfnSize, hiddenSize);
        Ops.AddInPlace(hidden, down, hiddenSize, pool: null);
    }

    private void DecodeGateUp(float* input, float* gate, float* up, int layer)
    {
        Weight wg = _weights[$"blk.{layer}.ffn_gate.weight"];
        Weight wu = _weights[$"blk.{layer}.ffn_up.weight"];
        int hidden = Config.HiddenSize;
        int ffn = Config.FfnSize;
        nuint q8Bytes = (nuint)Q8K.RowBytes(hidden);
        BlockQ8K* y = (BlockQ8K*)_gemmScratch.D(q8Bytes);
        Q8K.QuantizeRow(input, y, hidden);
        GemvPrequant(wg, y, gate, hidden, ffn);
        GemvPrequant(wu, y, up, hidden, ffn);
    }

    private void DecodeAttention(float* input, float* output, int layer, int start)
    {
        const int seq = 1;
        int heads = Config.NumHeads;
        int kvHeads = Config.NumKvHeads;
        int dim = Config.HeadDim;
        int qDim = heads * dim;
        int kDim = kvHeads * dim;

        // q_proj output is interleaved per-head: [q_h0, gate_h0, q_h1, gate_h1, ...].
        // Unpack into two [qDim] buffers (head-major) after the GEMV.
        float* qGateBuf = (float*)Bump((nuint)((long)2 * qDim * sizeof(float)));
        float* q = (float*)Bump((nuint)qDim * sizeof(float));
        float* gate = (float*)Bump((nuint)qDim * sizeof(float));
        float* k = _cacheK[layer] + start * kDim;
        float* v = _cacheV[layer] + start * kDim;

        nuint q8Bytes = (nuint)Q8K.RowBytes(Config.HiddenSize);
        BlockQ8K* y = (BlockQ8K*)_gemmScratch.D(q8Bytes);
        Q8K.QuantizeRow(input, y, Config.HiddenSize);
        Weight wq = _weights[$"blk.{layer}.attn_q.weight"];
        Weight wk = _weights[$"blk.{layer}.attn_k.weight"];
        Weight wv = _weights[$"blk.{layer}.attn_v.weight"];
        GemvPrequant(wq, y, qGateBuf, Config.HiddenSize, 2 * qDim);
        GemvPrequant(wk, y, k, Config.HiddenSize, kDim);
        GemvPrequant(wv, y, v, Config.HiddenSize, kDim);

        // Unpack interleaved q|gate into head-major layout.
        for (int h = 0; h < heads; h++)
        {
            int baseOff = h * 2 * dim;
            for (int d = 0; d < dim; d++)
            {
                q[h * dim + d] = qGateBuf[baseOff + d];
                gate[h * dim + d] = qGateBuf[baseOff + dim + d];
            }
        }

        long tRope = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
        Ops.NeoXRoPE(q, seq, heads, dim, Config.RopeDim, start, Config.RopeBase, pool: null);
        Ops.NeoXRoPE(k, seq, kvHeads, dim, Config.RopeDim, start, Config.RopeBase, pool: null);
        if (ProfileEnabled)
            TicksRope += Stopwatch.GetTimestamp() - tRope;
        RmsSerial($"blk.{layer}.attn_q_norm.weight", q, q, heads, dim);
        RmsSerial($"blk.{layer}.attn_k_norm.weight", k, k, kvHeads, dim);

        int kvLen = start + seq;
        float scale = 1f / MathF.Sqrt(dim);
        float* sc = (float*)Bump((nuint)((long)heads * kvLen * sizeof(float)));
        Ops.AttentionScores(q, _cacheK[layer], sc, heads, kvHeads, dim, seq, kvLen, qDim, kDim, scale, start, _pool);
        Ops.SoftmaxCausal(sc, heads, seq, kvLen, start, _pool);
        float* ao = (float*)Bump((nuint)qDim * sizeof(float));
        Ops.AttentionCombine(_cacheV[layer], sc, ao, heads, kvHeads, dim, seq, kvLen, qDim, kDim, start, _pool);

        // Apply attention gate: out = out * sigmoid(gate).
        for (int i = 0; i < qDim; i++)
            ao[i] *= 1f / (1f + MathF.Exp(-gate[i]));

        DecodeLinear(ao, $"blk.{layer}.attn_output.weight", output, qDim, Config.HiddenSize);
    }

    private void DecodeLinearAttn(float* input, float* output, int layer, int start)
    {
        int hidden = Config.HiddenSize;
        int nHeads = Config.SsmTimeStepRank;
        int headDim = Config.SsmStateSize;
        int vHeadDim = Config.SsmStateSize;
        int keyDim = Config.LinearKeyDim;
        int valueDim = Config.LinearValueDim;
        int convDim = Config.LinearConvDim;

        // 1) Projections.
        float* qkv = (float*)Bump((nuint)convDim * sizeof(float));
        DecodeLinear(input, $"blk.{layer}.attn_qkv.weight", qkv, hidden, convDim);
        float* z = (float*)Bump((nuint)valueDim * sizeof(float));
        DecodeLinear(input, $"blk.{layer}.attn_gate.weight", z, hidden, valueDim);
        float* bOut = (float*)Bump((nuint)nHeads * sizeof(float));
        DecodeLinear(input, $"blk.{layer}.ssm_beta.weight", bOut, hidden, nHeads);
        float* aOut = (float*)Bump((nuint)nHeads * sizeof(float));
        DecodeLinear(input, $"blk.{layer}.ssm_alpha.weight", aOut, hidden, nHeads);

        // 2) Causal conv1d over the qkv row (single token).
        float* convOut = (float*)Bump((nuint)convDim * sizeof(float));
        GatedDeltaNetOps.CausalConv1dUpdate(qkv, ConvKernel(layer), Config.SsmConvKernel, convDim, _linConv[layer], convOut);

        // 3) Split into q, k, v.
        float* q = convOut;
        float* k = convOut + keyDim;
        float* v = convOut + keyDim * 2;

        // 4) beta = sigmoid(bOut); g = -exp(A_log) * softplus(a + dt_bias).
        float* betaP = (float*)Bump((nuint)nHeads * sizeof(float));
        Qwen35Ops.Sigmoid(bOut, betaP, nHeads);
        float* gP = (float*)Bump((nuint)nHeads * sizeof(float));
        float* aLog = WeightF32($"blk.{layer}.ssm_a");
        float* dtBias = WeightF32($"blk.{layer}.ssm_dt.bias");
        for (int i = 0; i < nHeads; i++)
            gP[i] = -MathF.Exp(aLog[i]) * Qwen35Ops.SoftplusScalar(aOut[i] + dtBias[i]);

        // 5) Recurrence step for one token.
        float* state = _linState[layer];
        float* out_ = (float*)Bump((nuint)((long)nHeads * vHeadDim * sizeof(float)));
        GatedDeltaNetOps.RecurrentGatedDeltaRule(
            q, k, v, gP, betaP, state,
            1, nHeads, headDim, vHeadDim, Config.Eps, out_, l2normQk: true);

        // 6) Gated RMSNorm: norm(out) * silu(z).
        float* gated = (float*)Bump((nuint)((long)nHeads * vHeadDim * sizeof(float)));
        Qwen35Ops.GatedRmsNorm(out_, WeightF32($"blk.{layer}.ssm_norm.weight"), z, gated, nHeads, vHeadDim, Config.Eps);

        // 7) out_proj: [valueDim -> hidden].
        DecodeLinear(gated, $"blk.{layer}.ssm_out.weight", output, valueDim, hidden);
    }
}
