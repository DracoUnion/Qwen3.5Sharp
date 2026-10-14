using System.Runtime.CompilerServices;

namespace Flygon.Model;

/// <summary>
/// Ops specific to the Gated-DeltaNet linear-attention block: depthwise causal
/// conv1d (SiLU), and the recurrent delta-rule step.
///
/// These ops work on row-major float arrays matching the llama.cpp tensor
/// layouts used in <c>llm_build_qwen35::build_layer_attn_linear</c>:
///
///   q/i,k/v:  [token, head, dim]
///   g/beta:   [token, head]
///   state:    [head, headDim, vHeadDim]
/// </summary>
public static unsafe class GatedDeltaNetOps
{
    public static float Silu(float x) => x / (1f + MathF.Exp(-x));

    /// <summary>
    /// Causal depthwise conv1d with SiLU. Prefill variant.
    /// <paramref name="input"/>: <c>[channels][seq]</c>.
    /// <paramref name="kernel"/>: <c>[kernelSize][channels]</c>.
    /// Left-pads by <c>kernelSize - 1</c>. No bias.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CausalConv1d(
        float* input,
        float* kernel,
        int kernelSize,
        int channels,
        int seq,
        float* out_)
    {
        for (int c = 0; c < channels; c++)
        {
            float* inCh = input + (long)c * seq;
            float* outCh = out_ + (long)c * seq;
            float* kCh = kernel + c;
            for (int t = 0; t < seq; t++)
            {
                float acc = 0f;
                for (int j = 0; j < kernelSize; j++)
                {
                    int prev = t - j;
                    if (prev < 0)
                        continue;
                    acc += kCh[(long)j * channels] * inCh[prev];
                }
                outCh[t] = Silu(acc);
            }
        }
    }

    /// <summary>
    /// Single-token causal conv1d update. <paramref name="convState"/> holds
    /// the last <c>kernelSize - 1</c> input values per channel, layout
    /// <c>[channels][stateLen]</c> (index 0 = oldest). After the call the
    /// state is advanced to hold the new last <c>kernelSize - 1</c> values.
    /// <paramref name="input"/> and <paramref name="out_"/> are length
    /// <c>channels</c> (one new token's column); result is SiLU.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CausalConv1dUpdate(
        float* input,
        float* kernel,
        int kernelSize,
        int channels,
        float* convState,
        float* out_)
    {
        int stateLen = kernelSize - 1;
        for (int c = 0; c < channels; c++)
        {
            float* kCh = kernel + c;
            float* cs = convState + (long)c * stateLen;
            // The state holds the last stateLen input values (index 0 = oldest).
            // Extend the window with the new input value and apply the kernel.
            int j = 0;
            float acc = 0f;
            for (; j < stateLen; j++)
                acc += kCh[(long)j * channels] * cs[j];
            acc += kCh[(long)stateLen * channels] * input[c];
            out_[c] = Silu(acc);

            // Shift state left by one: state[j] = state[j+1]; state[stateLen-1] = input[c]
            for (j = 0; j < stateLen - 1; j++)
                cs[j] = cs[j + 1];
            cs[stateLen - 1] = input[c];
        }
    }

    /// <summary>
    /// Gated-DeltaNet recurrence for a sequence of tokens.
    ///
    /// <paramref name="q"/>, <paramref name="k"/>: <c>[tokens][nHeads][headDim]</c>
    /// <paramref name="v"/>:                       <c>[tokens][nHeads][vHeadDim]</c>
    /// <paramref name="g"/>:                       <c>[tokens][nHeads]</c> (log decay, <= 0)
    /// <paramref name="beta"/>:                    <c>[tokens][nHeads]</c>
    /// <paramref name="state"/>:                   <c>[nHeads][headDim][vHeadDim]</c> (in place)
    /// <paramref name="out_"/>:                    <c>[tokens][nHeads][vHeadDim]</c>
    ///
    /// Q is always scaled by <c>1/sqrt(headDim)</c>; when <paramref name="l2normQk"/>
    /// is set (always true for Qwen3.5), Q and K are L2-normalized before use.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RecurrentGatedDeltaRule(
        float* q,
        float* k,
        float* v,
        float* g,
        float* beta,
        float* state,
        int tokens,
        int nHeads,
        int headDim,
        int vHeadDim,
        float eps,
        float* out_,
        bool l2normQk)
    {
        float scaleQ = 1f / MathF.Sqrt(headDim);

        for (int h = 0; h < nHeads; h++)
        {
            float* S = state + (long)h * headDim * vHeadDim;

            for (int i = 0; i < tokens; i++)
            {
                float gIh = g[(long)i * nHeads + h];
                float betaIh = beta[(long)i * nHeads + h];
                float decay = MathF.Exp(gIh);

                float* qH = q + (long)i * nHeads * headDim + h * headDim;
                float* kH = k + (long)i * nHeads * headDim + h * headDim;
                float* vH = v + (long)i * nHeads * vHeadDim + h * vHeadDim;
                float* oH = out_ + (long)i * nHeads * vHeadDim + h * vHeadDim;

                if (l2normQk)
                {
                    // L2-normalize q and k into stack buffers; q is also scaled.
                    float[] qN = new float[headDim];
                    float[] kN = new float[headDim];
                    float qn = 0f, kn = 0f;
                    for (int a = 0; a < headDim; a++)
                    {
                        qn += qH[a] * qH[a];
                        kn += kH[a] * kH[a];
                    }
                    float qScale = scaleQ / MathF.Sqrt(qn + eps);
                    float kScale = 1f / MathF.Sqrt(kn + eps);
                    for (int a = 0; a < headDim; a++)
                    {
                        qN[a] = qH[a] * qScale;
                        kN[a] = kH[a] * kScale;
                    }
                    fixed (float* qp = qN, kp = kN)
                        DeltaStep(S, qp, kp, vH, decay, betaIh, headDim, vHeadDim, oH);
                }
                else
                {
                    // No L2-norm: scale Q into a stack buffer; K used as-is.
                    float[] qS = new float[headDim];
                    for (int a = 0; a < headDim; a++)
                        qS[a] = qH[a] * scaleQ;
                    fixed (float* qp = qS)
                        DeltaStep(S, qp, kH, vH, decay, betaIh, headDim, vHeadDim, oH);
                }
            }
        }
    }

    /// <summary>
    /// One delta-rule step. <paramref name="qIn"/> is already scaled (and
    /// optionally L2-normalized); <paramref name="kIn"/> is L2-normalized when
    /// the caller passed <c>l2normQk=true</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DeltaStep(
        float* S,
        float* qIn,
        float* kIn,
        float* vIn,
        float decay,
        float beta,
        int headDim,
        int vHeadDim,
        float* out_)
    {
        // 1) Decay state.
        for (int a = 0; a < headDim; a++)
        {
            float* row = S + a * vHeadDim;
            for (int b = 0; b < vHeadDim; b++)
                row[b] *= decay;
        }

        // 2) Read: kvMem[b] = sum_a S[a, b] * kIn[a]
        float[] kvMem = new float[vHeadDim];
        for (int b = 0; b < vHeadDim; b++)
        {
            float acc = 0f;
            for (int a = 0; a < headDim; a++)
                acc += S[a * vHeadDim + b] * kIn[a];
            kvMem[b] = acc;
        }

        // 3) delta[b] = (vIn[b] - kvMem[b]) * beta
        // 4) S[a, b] += kIn[a] * delta[b]
        for (int a = 0; a < headDim; a++)
        {
            float kA = kIn[a];
            float* row = S + a * vHeadDim;
            for (int b = 0; b < vHeadDim; b++)
            {
                float delta = (vIn[b] - kvMem[b]) * beta;
                row[b] += kA * delta;
            }
        }

        // 5) out[b] = sum_a S[a, b] * qIn[a]
        for (int b = 0; b < vHeadDim; b++)
        {
            float acc = 0f;
            for (int a = 0; a < headDim; a++)
                acc += S[a * vHeadDim + b] * qIn[a];
            out_[b] = acc;
        }
    }
}
