namespace Flygon.Kernels;

/// <summary>
/// IQ4_XS: non-linear 4-bit, super-block 256, 136 bytes per super-block.
/// Layout (llama.cpp block_iq4_xs):
///   d         fp16 grid scale
///   scales_h  uint16 — high 2 bits of each of the 8 sub-block (32) scales
///   scales_l  4 bytes — low 4 bits of each sub-block scale (packed 2/byte)
///   qs        128 bytes — 4-bit indices into kvalues_iq4nl, low nibble first
/// Sub-block scale: ls = (scales_l[ib/2] >> 4*(ib%2)) & 15 | (scales_h >> 2*ib & 3) << 4
///                 dl = d * (ls - 32)
/// </summary>
public static unsafe class IQ4XS
{
    private static readonly int[] KValues =
    {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };

    private static void SubBlockScale(BlockIQ4XS* x, int ib, out float dl, out float d)
    {
        int ls = ((x->ScalesL[ib / 2] >> (4 * (ib % 2))) & 0xF)
              | (((x->ScalesH >> (2 * ib)) & 3) << 4);
        d = HalfBits.ToSingle(x->D);
        dl = d * (ls - 32);
    }

    public static void DequantizeRow(BlockIQ4XS* x, float* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            byte* qs = x[i].Qs;
            for (int ib = 0; ib < Qk.SuperBlock / 32; ib++)
            {
                SubBlockScale(&x[i], ib, out float dl, out _);
                // 32 values: low nibble of qs[j], then high nibble of qs[j].
                for (int j = 0; j < 16; j++)
                {
                    y[j] = dl * KValues[qs[j] & 0xF];
                    y[j + 16] = dl * KValues[qs[j] >> 4];
                }
                qs += 16;
                y += 32;
            }
        }
    }

    public static float Dot(BlockIQ4XS* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        float sumf = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* qs = x[i].Qs;
            sbyte* q8 = y[i].Qs;
            float d8 = y[i].D;
            for (int ib = 0; ib < Qk.SuperBlock / 32; ib++)
            {
                SubBlockScale(&x[i], ib, out float dl, out _);
                // 32 dequantized weights from qs[0..16): low nibble then high nibble.
                int acc = 0;
                sbyte* q8Sub = q8 + ib * 32;
                for (int j = 0; j < 16; j++)
                {
                    acc += q8Sub[j] * KValues[qs[j] & 0xF];
                    acc += q8Sub[j + 16] * KValues[qs[j] >> 4];
                }
                sumf += dl * acc * d8;
                qs += 16;
            }
        }
        return sumf;
    }

    public static void Gemv(BlockIQ4XS* weights, float* input, float* output, int nIn, int nOut, CpuThreadPool? pool = null)
    {
        int nb = nIn / Qk.SuperBlock;
        using NativeBuffer q8 = new((nuint)Q8K.RowBytes(nIn));
        Q8K.QuantizeRow(input, (BlockQ8K*)q8.Pointer, nIn);
        GemvPrequant(weights, (BlockQ8K*)q8.Pointer, output, nIn, nOut, pool);
    }

    public static void GemvPrequant(BlockIQ4XS* weights, BlockQ8K* y, float* output, int nIn, int nOut, CpuThreadPool? pool = null)
    {
        int nb = nIn / Qk.SuperBlock;
        void Body(int worker, int workers)
        {
            int begin = nOut * worker / workers;
            int end = nOut * (worker + 1) / workers;
            for (int row = begin; row < end; row++)
                output[row] = Dot(weights + row * nb, y, nIn);
        }

        if (pool == null)
            Body(0, 1);
        else
            pool.For(nOut, Body);
    }

    public static void Gemm(BlockIQ4XS* weights, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null)
    {
        for (int t = 0; t < tokens; t++)
            Gemv(weights, input + t * nIn, output + t * nOut, nIn, nOut, pool);
    }
}