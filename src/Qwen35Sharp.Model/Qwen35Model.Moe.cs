using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Flygon.Gguf;
using Flygon.Kernels;

namespace Flygon.Model;

public sealed unsafe partial class Qwen35Model
{
    // =========================================================================
    // MoE FFN (sparse), arch "qwen35moe".
    //
    //   router_logits = x @ ffn_gate_inp^T           [tokens, num_experts]
    //   probs = softmax(router_logits)
    //   top-k indices/weights -> normalize weights to sum 1
    //   out   = Σ_k expert_weight * Expert_k(silu(gate_k(x)) * up_k(x))
    //   shared = sigmoid(x @ ffn_gate_inp_shexp^T) * SharedFFN(x)
    //   out  += shared
    //
    // Per-layer expert tensors (3D, [nIn, nOut, num_experts]):
    //   blk.L.ffn_gate_exps.weight  / ffn_up_exps.weight
    //   blk.L.ffn_down_exps.weight
    // Shared expert (2D):
    //   blk.L.ffn_gate_shexp.weight / ffn_up_shexp.weight / ffn_down_shexp.weight
    //   blk.L.ffn_gate_inp_shexp.weight (F32 [hidden] -> shared gate logit)
    // Router: blk.L.ffn_gate_inp.weight (F32 [hidden, num_experts])
    // =========================================================================

    private void PrefillSparseMoe(float* input, float* output, int layer, int seq)
    {
        bool trace = Environment.GetEnvironmentVariable("Q35_TRACE") == "1";
        void T(string s) { if (trace) Console.Error.WriteLine($"[MOE L{layer}] {s}"); }
        int hidden = Config.HiddenSize;
        int nExperts = Config.NumExperts;
        int nUsed = Config.NumExpertsPerTok;
        int ffn = Config.MoeFfnSize;
        int rows = seq;

        // 1) Router logits: [rows, nExperts].
        float* routerLogits = (float*)Bump((nuint)((long)rows * nExperts * sizeof(float)));
        Linear(_weights[$"blk.{layer}.ffn_gate_inp.weight"], input, routerLogits, hidden, nExperts, rows);
        T("router ok");

        // 2) Softmax + top-k.
        int* indices = (int*)Bump((nuint)((long)rows * nUsed * sizeof(int)));
        float* weights = (float*)Bump((nuint)((long)rows * nUsed * sizeof(float)));
        TopKSoftmax(routerLogits, rows, nExperts, nUsed, indices, weights);
        T("topk ok");

        // 3) Gate/up per selected expert, SiLU, down, weighted sum.
        for (int r = 0; r < rows; r++)
        {
            float* x = input + (long)r * hidden;
            for (int k = 0; k < nUsed; k++)
            {
                int exp = indices[r * nUsed + k];
                float w = weights[r * nUsed + k];
                // silu(gate) * up via two GEMVs per expert.
                float* gate = (float*)Bump((nuint)(ffn * sizeof(float)));
                float* up = (float*)Bump((nuint)(ffn * sizeof(float)));
                ExpertGateUp(x, layer, exp, gate, up, hidden, ffn);
                Ops.SiLUMul(gate, up, ffn, pool: null);
                float* down = (float*)Bump((nuint)(hidden * sizeof(float)));
                ExpertDown(gate, layer, exp, down, ffn, hidden);
                // output[r] += w * down
                float* o = output + (long)r * hidden;
                for (int d = 0; d < hidden; d++)
                    o[d] += w * down[d];
            }
        }
        T("experts ok");

        // 4) Shared expert (if present). llama.cpp gates on tensor presence.
        if (_weights.ContainsKey($"blk.{layer}.ffn_gate_shexp.weight"))
        {
            // shared = sigmoid(x @ ffn_gate_inp_shexp^T) * SharedFFN(x)
            float* sharedGateLogit = (float*)Bump((nuint)(rows * sizeof(float)));
            F32Gemm(_weights[$"blk.{layer}.ffn_gate_inp_shexp.weight"].F32, input, sharedGateLogit, hidden, 1, rows);

            // Shared FFN: gate_shexp/up_shexp/down_shexp, no per-token expert loop.
            float* sharedGate = (float*)Bump((nuint)((long)rows * Config.MoeSharedFfnSize * sizeof(float)));
            float* sharedUp = (float*)Bump((nuint)((long)rows * Config.MoeSharedFfnSize * sizeof(float)));
            Linear(_weights[$"blk.{layer}.ffn_gate_shexp.weight"], input, sharedGate, hidden, Config.MoeSharedFfnSize, rows);
            Linear(_weights[$"blk.{layer}.ffn_up_shexp.weight"], input, sharedUp, hidden, Config.MoeSharedFfnSize, rows);
            Ops.SiLUMul(sharedGate, sharedUp, checked((int)((long)rows * Config.MoeSharedFfnSize)), _pool);
            float* sharedOut = (float*)Bump((nuint)((long)rows * hidden * sizeof(float)));
            Linear(_weights[$"blk.{layer}.ffn_down_shexp.weight"], sharedGate, sharedOut, Config.MoeSharedFfnSize, hidden, rows);

            for (int r = 0; r < rows; r++)
            {
                float sg = 1f / (1f + MathF.Exp(-sharedGateLogit[r]));
                float* o = output + (long)r * hidden;
                float* s = sharedOut + (long)r * hidden;
                for (int d = 0; d < hidden; d++)
                    o[d] += sg * s[d];
            }
        }
    }

    private void DecodeSparseMoe(float* input, float* output, int layer)
    {
        const int rows = 1;
        int hidden = Config.HiddenSize;
        int nExperts = Config.NumExperts;
        int nUsed = Config.NumExpertsPerTok;
        int ffn = Config.MoeFfnSize;

        float* routerLogits = (float*)Bump((nuint)(nExperts * sizeof(float)));
        Linear(_weights[$"blk.{layer}.ffn_gate_inp.weight"], input, routerLogits, hidden, nExperts, 1);

        int* indices = (int*)Bump((nuint)(nUsed * sizeof(int)));
        float* weights = (float*)Bump((nuint)(nUsed * sizeof(float)));
        TopKSoftmax(routerLogits, rows, nExperts, nUsed, indices, weights);

        float* x = input;
        for (int k = 0; k < nUsed; k++)
        {
            int exp = indices[k];
            float w = weights[k];
            float* gate = (float*)Bump((nuint)(ffn * sizeof(float)));
            float* up = (float*)Bump((nuint)(ffn * sizeof(float)));
            ExpertGateUp(x, layer, exp, gate, up, hidden, ffn);
            Ops.SiLUMul(gate, up, ffn, pool: null);
            float* down = (float*)Bump((nuint)(hidden * sizeof(float)));
            ExpertDown(gate, layer, exp, down, ffn, hidden);
            for (int d = 0; d < hidden; d++)
                output[d] += w * down[d];
        }

        if (_weights.ContainsKey($"blk.{layer}.ffn_gate_shexp.weight"))
        {
            float* sharedGateLogit = (float*)Bump((nuint)sizeof(float));
            F32Gemm(_weights[$"blk.{layer}.ffn_gate_inp_shexp.weight"].F32, input, sharedGateLogit, hidden, 1, 1);
            float* sharedGate = (float*)Bump((nuint)(Config.MoeSharedFfnSize * sizeof(float)));
            float* sharedUp = (float*)Bump((nuint)(Config.MoeSharedFfnSize * sizeof(float)));
            DecodeLinear(_weights[$"blk.{layer}.ffn_gate_shexp.weight"], input, sharedGate, hidden, Config.MoeSharedFfnSize);
            DecodeLinear(_weights[$"blk.{layer}.ffn_up_shexp.weight"], input, sharedUp, hidden, Config.MoeSharedFfnSize);
            Ops.SiLUMul(sharedGate, sharedUp, Config.MoeSharedFfnSize, pool: null);
            float* sharedOut = (float*)Bump((nuint)(hidden * sizeof(float)));
            DecodeLinear(_weights[$"blk.{layer}.ffn_down_shexp.weight"], sharedGate, sharedOut, Config.MoeSharedFfnSize, hidden);
            float sg = 1f / (1f + MathF.Exp(-sharedGateLogit[0]));
            for (int d = 0; d < hidden; d++)
                output[d] += sg * sharedOut[d];
        }
    }

    /// <summary>
    /// softmax(x) then pick the top-k entries per row; renormalize the top-k
    /// weights to sum to 1. Writes indices (descending by weight) and weights.
    /// </summary>
    private void TopKSoftmax(float* logits, int rows, int nExperts, int nUsed, int* indices, float* weights)
    {
        for (int r = 0; r < rows; r++)
        {
            float* row = logits + (long)r * nExperts;
            // softmax
            float max = row[0];
            for (int i = 1; i < nExperts; i++)
                if (row[i] > max)
                    max = row[i];
            float sum = 0;
            for (int i = 0; i < nExperts; i++)
            {
                row[i] = MathF.Exp(row[i] - max);
                sum += row[i];
            }
            float inv = 1f / sum;
            for (int i = 0; i < nExperts; i++)
                row[i] *= inv;

            // top-k by selection (nUsed <= 8, nExperts ~256; simple scan is fine).
            int* idx = indices + (long)r * nUsed;
            float* wgt = weights + (long)r * nUsed;
            for (int k = 0; k < nUsed; k++)
            {
                int best = -1;
                float bestV = -1f;
                for (int i = 0; i < nExperts; i++)
                {
                    if (row[i] > bestV)
                    {
                        bestV = row[i];
                        best = i;
                    }
                }
                idx[k] = best;
                wgt[k] = bestV;
                row[best] = -1f; // exclude for next pass
            }
            // Renormalize the selected weights.
            float wsum = 0;
            for (int k = 0; k < nUsed; k++)
                wsum += wgt[k];
            float winv = wsum > 0 ? 1f / wsum : 0f;
            for (int k = 0; k < nUsed; k++)
                wgt[k] *= winv;
        }
    }

    /// <summary>
    /// gate/up = x @ Expert_k(gate|up)^T for one expert. The weight is a 3D
    /// tensor with nIn innermost; the expert's slice is at a byte stride.
    /// </summary>
    private void ExpertGateUp(float* x, int layer, int exp, float* gate, float* up, int hidden, int ffn)
    {
        Weight wg = _weights[$"blk.{layer}.ffn_gate_exps.weight"];
        Weight wu = _weights[$"blk.{layer}.ffn_up_exps.weight"];
        byte* gBase = ExpertBase(wg, exp);
        byte* uBase = ExpertBase(wu, exp);
        switch (wg.Type)
        {
            case GgmlTensorType.IQ4_XS:
                IQ4XS.Gemv((BlockIQ4XS*)gBase, x, gate, hidden, ffn, _pool);
                IQ4XS.Gemv((BlockIQ4XS*)uBase, x, up, hidden, ffn, _pool);
                break;
            case GgmlTensorType.Q5_K:
                Q5K.Gemv((BlockQ5K*)gBase, x, gate, hidden, ffn, _pool);
                Q5K.Gemv((BlockQ5K*)uBase, x, up, hidden, ffn, _pool);
                break;
            case GgmlTensorType.Q4_K:
                MulMatQ4K.Gemv((BlockQ4K*)gBase, x, gate, hidden, ffn, _pool, _gemmScratch);
                MulMatQ4K.Gemv((BlockQ4K*)uBase, x, up, hidden, ffn, _pool, _gemmScratch);
                break;
            default:
                throw new NotSupportedException($"{wg.Type} expert weight");
        }
    }

    private void ExpertDown(float* gate, int layer, int exp, float* down, int ffn, int hidden)
    {
        Weight wd = _weights[$"blk.{layer}.ffn_down_exps.weight"];
        byte* dBase = ExpertBase(wd, exp);
        switch (wd.Type)
        {
            case GgmlTensorType.IQ4_XS:
                IQ4XS.Gemv((BlockIQ4XS*)dBase, gate, down, ffn, hidden, _pool);
                break;
            case GgmlTensorType.Q5_K:
                Q5K.Gemv((BlockQ5K*)dBase, gate, down, ffn, hidden, _pool);
                break;
            case GgmlTensorType.Q4_K:
                MulMatQ4K.Gemv((BlockQ4K*)dBase, gate, down, ffn, hidden, _pool, _gemmScratch);
                break;
            default:
                throw new NotSupportedException($"{wd.Type} expert down weight");
        }
    }

    /// <summary>Base pointer of one expert slice, offset by type-specific pointer.</summary>
    private static byte* ExpertBase(Weight w, int exp)
    {
        byte* p = w.Type switch
        {
            GgmlTensorType.IQ4_XS => (byte*)w.Q4xs,
            GgmlTensorType.Q5_K => (byte*)w.Q5,
            GgmlTensorType.Q4_K => (byte*)w.Q4,
            _ => throw new NotSupportedException($"{w.Type} expert weight"),
        };
        return p + exp * w.ExpertStride;
    }
}