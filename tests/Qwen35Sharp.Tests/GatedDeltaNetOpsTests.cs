using System.Runtime.InteropServices;
using Flygon.Model;
using Flygon.Kernels;
using Xunit;

namespace Flygon.Qwen35Sharp.Tests;

public sealed class GatedDeltaNetOpsTests
{
    // Layout helpers match GatedDeltaNetOps.RecurrentGatedDeltaRule:
    //   q/k [token, head, headDim]; v [token, head, vHeadDim]
    //   g/beta [token, head]; state [head, headDim, vHeadDim]
    private static int qkIndex(int t, int h, int nHeads, int headDim) => (t * nHeads + h) * headDim;
    private static int vIndex(int t, int h, int nHeads, int vHeadDim) => (t * nHeads + h) * vHeadDim;
    private static int sIndex(int h, int a, int b, int headDim, int vHeadDim) => h * headDim * vHeadDim + a * vHeadDim + b;

    [Fact]
    public unsafe void RecurrentGatedDeltaRule_MatchesReference()
    {
        const int tokens = 5, nHeads = 2, headDim = 4, vHeadDim = 4;
        const float eps = 1e-6f;
        Random rng = new(1234);

        float[] q = new float[tokens * nHeads * headDim];
        float[] k = new float[tokens * nHeads * headDim];
        float[] v = new float[tokens * nHeads * vHeadDim];
        float[] g = new float[tokens * nHeads];
        float[] beta = new float[tokens * nHeads];
        float[] state = new float[nHeads * headDim * vHeadDim];

        FillRandom(rng, q);
        FillRandom(rng, k);
        FillRandom(rng, v);
        for (int i = 0; i < g.Length; i++)
            g[i] = -(float)Math.Abs(rng.NextDouble()) - 0.01f; // g <= 0
        for (int i = 0; i < beta.Length; i++)
            beta[i] = (float)(0.5 * rng.NextDouble());

        // ---- Independent reference (no L2-normalization variant is the Qwen3.5
        // ---- case: l2normQk=true). Mirror the exact math.
        float scaleQ = 1f / MathF.Sqrt(headDim);
        float[] refState = new float[state.Length];
        float[] refOut = new float[tokens * nHeads * vHeadDim];

        for (int h = 0; h < nHeads; h++)
        {
            for (int t = 0; t < tokens; t++)
            {
                int qBase = qkIndex(t, h, nHeads, headDim);
                int kB = qkIndex(t, h, nHeads, headDim);
                int vB = vIndex(t, h, nHeads, vHeadDim);
                float decay = MathF.Exp(g[t * nHeads + h]);
                float bh = beta[t * nHeads + h];

                float qn = 0f, kn = 0f;
                for (int a = 0; a < headDim; a++)
                {
                    qn += q[qBase + a] * q[qBase + a];
                    kn += k[kB + a] * k[kB + a];
                }
                float qScale = scaleQ / MathF.Sqrt(qn + eps);
                float kScale = 1f / MathF.Sqrt(kn + eps);

                // decay state
                for (int a = 0; a < headDim; a++)
                    for (int b = 0; b < vHeadDim; b++)
                        refState[sIndex(h, a, b, headDim, vHeadDim)] *= decay;

                float[] kvMem = new float[vHeadDim];
                for (int b = 0; b < vHeadDim; b++)
                {
                    float acc = 0f;
                    for (int a = 0; a < headDim; a++)
                        acc += refState[sIndex(h, a, b, headDim, vHeadDim)] * (k[kB + a] * kScale);
                    kvMem[b] = acc;
                }
                for (int a = 0; a < headDim; a++)
                {
                    float kA = k[kB + a] * kScale;
                    for (int b = 0; b < vHeadDim; b++)
                    {
                        float delta = (v[vB + b] - kvMem[b]) * bh;
                        refState[sIndex(h, a, b, headDim, vHeadDim)] += kA * delta;
                    }
                }
                for (int b = 0; b < vHeadDim; b++)
                {
                    float acc = 0f;
                    for (int a = 0; a < headDim; a++)
                        acc += refState[sIndex(h, a, b, headDim, vHeadDim)] * (q[qBase + a] * qScale);
                    refOut[vIndex(t, h, nHeads, vHeadDim) + b] = acc;
                }
            }
        }

        // ---- SUT
        float[] stateCopy = new float[state.Length];
        float[] gotOut = new float[refOut.Length];
        fixed (float* qP = q, kP = k, vP = v, gP = g, betaP = beta,
                 sP = stateCopy, oP = gotOut)
        {
            GatedDeltaNetOps.RecurrentGatedDeltaRule(
                qP, kP, vP, gP, betaP, sP,
                tokens, nHeads, headDim, vHeadDim, eps, oP, l2normQk: true);
        }

        for (int i = 0; i < refOut.Length; i++)
            Assert.True(Math.Abs(refOut[i] - gotOut[i]) < 1e-3,
                $"out[{i}]: ref={refOut[i]} got={gotOut[i]}");
        for (int i = 0; i < refState.Length; i++)
            Assert.True(Math.Abs(refState[i] - stateCopy[i]) < 1e-3,
                $"state[{i}]: ref={refState[i]} got={stateCopy[i]}");
    }

    private static void FillRandom(Random r, float[] a)
    {
        for (int i = 0; i < a.Length; i++)
            a[i] = (float)(r.NextDouble() * 4 - 2);
    }
}
