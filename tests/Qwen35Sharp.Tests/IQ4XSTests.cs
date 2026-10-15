using Flygon.Kernels;
using Xunit;

namespace Flygon.Qwen35Sharp.Tests;

public sealed class IQ4XSTests
{
    private static readonly int[] KValues =
    {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };

    [Fact]
    public unsafe void DequantizeRow_MatchesReference()
    {
        const int k = 256 * 3; // 3 super-blocks
        int nb = k / 256;
        Random rng = new(7);

        var raw = new byte[nb * Qk.IQ4XSSize];
        rng.NextBytes(raw);
        ForceValidScales(raw, nb); // avoid random NaN/NegZero fp16 grid scales

        fixed (byte* p = raw)
        {
            BlockIQ4XS* x = (BlockIQ4XS*)p;
            // Reference dequant per llama.cpp dequantize_row_iq4_xs.
            float[] expected = new float[k];
            float[] got = new float[k];
            for (int i = 0; i < nb; i++)
            {
                byte* qs = x[i].Qs;
                float d = HalfBits.ToSingle(x[i].D);
                for (int ib = 0; ib < 8; ib++)
                {
                    int ls = ((x[i].ScalesL[ib / 2] >> (4 * (ib % 2))) & 0xF)
                           | (((x[i].ScalesH >> (2 * ib)) & 3) << 4);
                    float dl = d * (ls - 32);
                    for (int j = 0; j < 16; j++)
                    {
                        expected[i * 256 + ib * 32 + j] = dl * KValues[qs[j] & 0xF];
                        expected[i * 256 + ib * 32 + j + 16] = dl * KValues[qs[j] >> 4];
                    }
                    qs += 16;
                }
            }

            fixed (float* g = got)
                IQ4XS.DequantizeRow(x, g, k);

            for (int i = 0; i < k; i++)
                Assert.True(MathF.Abs(expected[i] - got[i]) < 1e-4f,
                    $"dequant[{i}]: ref={expected[i]} got={got[i]}");
        }
    }

    [Fact]
    public unsafe void Dot_MatchesScalarDequant()
    {
        const int k = 256 * 2;
        int nb = k / 256;
        Random rng = new(9);

        var raw = new byte[nb * Qk.IQ4XSSize];
        rng.NextBytes(raw);
        ForceValidScales(raw, nb);

        // Build a Q8K block directly.
        var q8Raw = new byte[nb * Qk.Q8KSize];
        float[] act = new float[k];
        for (int i = 0; i < k; i++)
            act[i] = (float)(rng.NextDouble() * 4 - 2);
        fixed (byte* q8p = q8Raw)
        fixed (float* actP = act)
        {
            Q8K.QuantizeRow(actP, (BlockQ8K*)q8p, k);
            BlockQ8K* y = (BlockQ8K*)q8p;
            fixed (byte* p = raw)
            {
                BlockIQ4XS* x = (BlockIQ4XS*)p;
                float dot = IQ4XS.Dot(x, y, k);

                // Reference: dequant IQ4 then dot against dequant Q8.
                float[] iq = new float[k];
                float[] q8 = new float[k];
                fixed (float* ip = iq)
                    IQ4XS.DequantizeRow(x, ip, k);
                for (int i = 0; i < nb; i++)
                {
                    for (int j = 0; j < 256; j++)
                        q8[i * 256 + j] = y[i].D * y[i].Qs[j];
                }
                float expected = 0;
                for (int i = 0; i < k; i++)
                    expected += iq[i] * q8[i];

                float err = MathF.Abs(expected - dot);
                Assert.True(err < 1e-2f || err < 1e-4f * MathF.Abs(expected), $"dot: ref={expected} got={dot}");
            }
        }
    }

    /// <summary>Force the per-block fp16 grid scale to 1.0f to avoid random NaN/Inf.</summary>
    private static void ForceValidScales(byte[] raw, int nb)
    {
        for (int i = 0; i < nb; i++)
        {
            int off = i * Qk.IQ4XSSize;
            raw[off + 0] = 0x00;
            raw[off + 1] = 0x3C; // fp16 1.0f
        }
    }
}
