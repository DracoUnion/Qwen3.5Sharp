using System.Diagnostics;
using Flygon.Gguf;
using Flygon.Kernels;

namespace Flygon.Model;

public sealed unsafe partial class Qwen35Model
{
    // =========================================================================
    // Prefill (N tokens)
    // =========================================================================

    private void PrefillBlock(float* hidden, int layer, int seq, int start)
    {
        int hiddenSize = Config.HiddenSize;
        float* n1 = (float*)Bump((nuint)((long)seq * hiddenSize * sizeof(float)));
        float* attn = (float*)Bump((nuint)((long)seq * hiddenSize * sizeof(float)));
        Rms($"blk.{layer}.attn_norm.weight", hidden, n1, seq, hiddenSize);

        if (Config.IsRecurrent[layer])
            PrefillLinearAttn(n1, attn, layer, seq, start);
        else
            PrefillAttention(n1, attn, layer, seq, start);
        Ops.AddInPlace(hidden, attn, checked((int)((long)seq * hiddenSize)), _pool);

        float* n2 = (float*)Bump((nuint)((long)seq * hiddenSize * sizeof(float)));
        Rms($"blk.{layer}.post_attention_norm.weight", hidden, n2, seq, hiddenSize);
        float* gate = (float*)Bump((nuint)((long)seq * Config.FfnSize * sizeof(float)));
        float* up = (float*)Bump((nuint)((long)seq * Config.FfnSize * sizeof(float)));
        float* down = (float*)Bump((nuint)((long)seq * hiddenSize * sizeof(float)));
        PrefillGateUp(n2, gate, up, layer, seq);
        Ops.SiLUMul(gate, up, checked((int)((long)seq * Config.FfnSize)), _pool);
        Linear(gate, $"blk.{layer}.ffn_down.weight", down, Config.FfnSize, hiddenSize, seq);
        Ops.AddInPlace(hidden, down, checked((int)((long)seq * hiddenSize)), _pool);
    }

    private void PrefillGateUp(float* input, float* gate, float* up, int layer, int seq)
    {
        Weight wg = _weights[$"blk.{layer}.ffn_gate.weight"];
        Weight wu = _weights[$"blk.{layer}.ffn_up.weight"];
        int hidden = Config.HiddenSize;
        int ffn = Config.FfnSize;
        switch (wg.Type)
        {
            case GgmlTensorType.Q2_0C:
                MulMatQ2.Gemm(wg.Packed2, wg.Q2, input, gate, hidden, ffn, seq, _pool, _gemmScratch);
                MulMatQ2.Gemm(wu.Packed2, wu.Q2, input, up, hidden, ffn, seq, _pool, _gemmScratch);
                break;
            case GgmlTensorType.STQ1_0:
                MulMatSTQ.Gemm(wg.PackedSTQ, wg.STQ, input, gate, hidden, ffn, seq, _pool, _gemmScratch);
                MulMatSTQ.Gemm(wu.PackedSTQ, wu.STQ, input, up, hidden, ffn, seq, _pool, _gemmScratch);
                break;
            case GgmlTensorType.Q4_K:
                MulMatQ4K.Gemm(wg.Packed, wg.Q4, input, gate, hidden, ffn, seq, _pool, _gemmScratch, wg.Meta);
                MulMatQ4K.Gemm(wu.Packed, wu.Q4, input, up, hidden, ffn, seq, _pool, _gemmScratch, wu.Meta);
                break;
            case GgmlTensorType.Q5_K:
                Q5K.Gemm(wg.Q5, input, gate, hidden, ffn, seq, _pool);
                Q5K.Gemm(wu.Q5, input, up, hidden, ffn, seq, _pool);
                break;
            case GgmlTensorType.Q6_K:
                Q6K.Gemm(wg.Q6, input, gate, hidden, ffn, seq, _pool, _gemmScratch);
                Q6K.Gemm(wu.Q6, input, up, hidden, ffn, seq, _pool, _gemmScratch);
                break;
            default:
                throw new NotSupportedException($"{wg.Type} not a GEMM weight");
        }
    }

    private void PrefillAttention(float* input, float* output, int layer, int seq, int start)
    {
        bool trace = Environment.GetEnvironmentVariable("Q35_TRACE") == "1";
        void T(string s) { if (trace) Console.Error.WriteLine($"[FA L{layer}] {s}"); }
        int heads = Config.NumHeads;
        int kvHeads = Config.NumKvHeads;
        int dim = Config.HeadDim;
        int qDim = heads * dim;
        int kDim = kvHeads * dim;
        int hidden = Config.HiddenSize;
        T($"seq={seq} start={start} heads={heads} kvHeads={kvHeads} dim={dim} qDim={qDim} kDim={kDim}");

        // Q + gate: GEMM output qGateBuf is [seq][2*qDim] where the feature
        // dim is interleaved per-head as [q_h0, gate_h0, q_h1, gate_h1, ...].
        // Unpack into two [seq][qDim] buffers with head-major order after the GEMM.
        float* qGateBuf = (float*)Bump((nuint)((long)seq * 2 * qDim * sizeof(float)));
        float* q = (float*)Bump((nuint)((long)seq * qDim * sizeof(float)));
        float* gate = (float*)Bump((nuint)((long)seq * qDim * sizeof(float)));
        float* k = _cacheK[layer] + start * kDim;
        float* v = _cacheV[layer] + start * kDim;

        Linear(input, $"blk.{layer}.attn_q.weight", qGateBuf, hidden, 2 * qDim, seq);
        T("gemm q ok");
        // Unpack the interleaved q|gate into head-major [seq][heads][dim].
        for (int t = 0; t < seq; t++)
        {
            float* row = qGateBuf + (long)t * 2 * qDim;
            float* qRow = q + (long)t * qDim;
            float* gRow = gate + (long)t * qDim;
            for (int h = 0; h < heads; h++)
            {
                int baseOff = h * 2 * dim;
                for (int d = 0; d < dim; d++)
                {
                    qRow[h * dim + d] = row[baseOff + d];
                    gRow[h * dim + d] = row[baseOff + dim + d];
                }
            }
        }
        T("unpack q/gate ok");
        Linear(input, $"blk.{layer}.attn_k.weight", k, hidden, kDim, seq);
        Linear(input, $"blk.{layer}.attn_v.weight", v, hidden, kDim, seq);

        Ops.NeoXRoPE(q, seq, heads, dim, Config.RopeDim, start, Config.RopeBase, _pool);
        Ops.NeoXRoPE(k, seq, kvHeads, dim, Config.RopeDim, start, Config.RopeBase, _pool);
        Rms($"blk.{layer}.attn_q_norm.weight", q, q, seq * heads, dim);
        Rms($"blk.{layer}.attn_k_norm.weight", k, k, seq * kvHeads, dim);

        int kvLen = start + seq;
        float scale = 1f / MathF.Sqrt(dim);
        float* sc = (float*)Bump((nuint)((long)heads * seq * kvLen * sizeof(float)));
        Ops.AttentionScores(q, _cacheK[layer], sc, heads, kvHeads, dim, seq, kvLen, qDim, kDim, scale, start, _pool);
        Ops.SoftmaxCausal(sc, heads, seq, kvLen, start, _pool);
        float* ao = (float*)Bump((nuint)((long)seq * qDim * sizeof(float)));
        Ops.AttentionCombine(_cacheV[layer], sc, ao, heads, kvHeads, dim, seq, kvLen, qDim, kDim, start, _pool);

        // Apply gate over all seq rows.
        int total = seq * qDim;
        for (int i = 0; i < total; i++)
            ao[i] *= 1f / (1f + MathF.Exp(-gate[i]));

        Linear(ao, $"blk.{layer}.attn_output.weight", output, qDim, hidden, seq);
    }

    private void PrefillLinearAttn(float* input, float* output, int layer, int seq, int start)
    {
        bool trace = Environment.GetEnvironmentVariable("Q35_TRACE") == "1";
        void T(string s) { if (trace) Console.Error.WriteLine($"[L{layer}] {s}"); }
        int hidden = Config.HiddenSize;
        int nHeads = Config.SsmTimeStepRank;
        int headDim = Config.SsmStateSize;
        int vHeadDim = Config.SsmStateSize;
        int keyDim = Config.LinearKeyDim;
        int valueDim = Config.LinearValueDim;
        int convDim = Config.LinearConvDim;
        T($"prefill linear seq={seq} convDim={convDim} keyDim={keyDim} valueDim={valueDim} nHeads={nHeads} headDim={headDim}");

        // 1) qkv: [seq, convDim]
        float* qkv = (float*)Bump((nuint)((long)seq * convDim * sizeof(float)));
        Linear(input, $"blk.{layer}.attn_qkv.weight", qkv, hidden, convDim, seq);
        T("qkv ok");
        // z: [seq, valueDim]
        float* z = (float*)Bump((nuint)((long)seq * valueDim * sizeof(float)));
        Linear(input, $"blk.{layer}.attn_gate.weight", z, hidden, valueDim, seq);
        // b: [seq, nHeads]
        float* bOut = (float*)Bump((nuint)((long)seq * nHeads * sizeof(float)));
        Linear(input, $"blk.{layer}.ssm_beta.weight", bOut, hidden, nHeads, seq);
        // a: [seq, nHeads]
        float* aOut = (float*)Bump((nuint)((long)seq * nHeads * sizeof(float)));
        Linear(input, $"blk.{layer}.ssm_alpha.weight", aOut, hidden, nHeads, seq);

        // 2) Causal conv1d over the full sequence.
        // Input layout for CausalConv1d: [channels][seq] (channel-major).
        // Our qkv is [seq][convDim] (seq-major). Transpose.
        float* qkvCh = (float*)Bump((nuint)((long)convDim * seq * sizeof(float)));
        for (int c = 0; c < convDim; c++)
            for (int t = 0; t < seq; t++)
                qkvCh[(long)c * seq + t] = qkv[(long)t * convDim + c];
        float* convOutCh = (float*)Bump((nuint)((long)convDim * seq * sizeof(float)));
        GatedDeltaNetOps.CausalConv1d(qkvCh, ConvKernel(layer), Config.SsmConvKernel, convDim, seq, convOutCh);
        T("conv ok");

        // Update the conv state for the next call: the conv state holds the
        // last (kernelSize - 1) *raw input* values per channel (index 0 =
        // oldest), matching the decode path's CausalConv1dUpdate.
        int stateLen = Math.Max(1, Config.SsmConvKernel - 1);
        for (int c = 0; c < convDim; c++)
        {
            float* cs = _linConv[layer] + (long)c * stateLen;
            int tailStart = seq - stateLen;
            if (tailStart < 0)
                tailStart = 0;
            for (int j = 0; j < stateLen; j++)
                cs[j] = qkvCh[(long)c * seq + tailStart + j];
        }

        // 3) Split into q, k, v (seq-major): q [seq, keyDim], k [seq, keyDim],
        //    v [seq, valueDim].
        float* q = convOutCh; // not actually seq-major; we need to un-transpose.
        // Build seq-major q, k, v.
        float* qSm = (float*)Bump((nuint)((long)seq * keyDim * sizeof(float)));
        float* kSm = (float*)Bump((nuint)((long)seq * keyDim * sizeof(float)));
        float* vSm = (float*)Bump((nuint)((long)seq * valueDim * sizeof(float)));
        for (int t = 0; t < seq; t++)
        {
            for (int d = 0; d < keyDim; d++)
                qSm[(long)t * keyDim + d] = convOutCh[(long)d * seq + t];
            for (int d = 0; d < keyDim; d++)
                kSm[(long)t * keyDim + d] = convOutCh[(long)(keyDim + d) * seq + t];
            for (int d = 0; d < valueDim; d++)
                vSm[(long)t * valueDim + d] = convOutCh[(long)(2 * keyDim + d) * seq + t];
        }

        // 4) beta = sigmoid(bOut), g = -exp(A_log) * softplus(a + dt_bias).
        float* betaP = (float*)Bump((nuint)((long)seq * nHeads * sizeof(float)));
        Qwen35Ops.Sigmoid(bOut, betaP, (int)((long)seq * nHeads));
        float* gP = (float*)Bump((nuint)((long)seq * nHeads * sizeof(float)));
        float* aLog = WeightF32($"blk.{layer}.ssm_a");
        float* dtBias = WeightF32($"blk.{layer}.ssm_dt.bias");
        for (int i = 0; i < seq * nHeads; i++)
        {
            int head = i % nHeads;
            gP[i] = -MathF.Exp(aLog[head]) * Qwen35Ops.SoftplusScalar(aOut[i] + dtBias[head]);
        }

        // 5) Recurrence for the whole sequence.
        float* state = _linState[layer];
        float* out_ = (float*)Bump((nuint)((long)seq * nHeads * vHeadDim * sizeof(float)));
        GatedDeltaNetOps.RecurrentGatedDeltaRule(
            qSm, kSm, vSm, gP, betaP, state,
            seq, nHeads, headDim, vHeadDim, Config.Eps, out_, l2normQk: true);
        T("recurrence ok");

        // 6) Gated RMSNorm: norm(out) * silu(z).
        float* gated = (float*)Bump((nuint)((long)seq * nHeads * vHeadDim * sizeof(float)));
        Qwen35Ops.GatedRmsNorm(out_, WeightF32($"blk.{layer}.ssm_norm.weight"), z, gated, seq * nHeads, vHeadDim, Config.Eps);
        T("gated norm ok");

        // 7) out_proj: [valueDim -> hidden] per row.
        Linear(gated, $"blk.{layer}.ssm_out.weight", output, valueDim, hidden, seq);
        T("out proj ok");
    }
}
