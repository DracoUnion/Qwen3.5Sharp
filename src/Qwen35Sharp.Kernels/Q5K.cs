namespace Flygon.Kernels;

public static unsafe class Q5K
{
    public static void DequantizeRow(BlockQ5K* x, float* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            byte* ql = x[i].Qs;
            byte* qh = x[i].Qh;
            float d = HalfBits.ToSingle(x[i].D);
            float min = HalfBits.ToSingle(x[i].Dmin);
            int iscale = 0;
            byte u1 = 1;
            byte u2 = 2;
            for (int j = 0; j < Qk.SuperBlock; j += 64)
            {
                Q4K.GetScaleMin(iscale, x[i].Scales, out byte sc0, out byte min0);
                Q4K.GetScaleMin(iscale + 1, x[i].Scales, out byte sc1, out byte min1);
                float d1 = d * sc0;
                float m1 = min * min0;
                float d2 = d * sc1;
                float m2 = min * min1;
                for (int l = 0; l < 32; l++)
                    *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; l++)
                    *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
                ql += 32;
                iscale += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    public static float Dot(BlockQ5K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        uint* utmp = stackalloc uint[4];
        sbyte* aux = stackalloc sbyte[Qk.SuperBlock];
        float sumf = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* q4 = x[i].Qs;
            byte* hm = x[i].Qh;
            sbyte* a = aux;
            byte m = 1;
            for (int j = 0; j < Qk.SuperBlock / 64; j++)
            {
                for (int l = 0; l < 32; l++)
                    a[l] = (sbyte)((q4[l] & 0xF) + ((hm[l] & m) != 0 ? 16 : 0));
                a += 32;
                m <<= 1;
                for (int l = 0; l < 32; l++)
                    a[l] = (sbyte)((q4[l] >> 4) + ((hm[l] & m) != 0 ? 16 : 0));
                a += 32;
                m <<= 1;
                q4 += 32;
            }

            Q4K.UnpackScales(x[i].Scales, utmp);
            byte* scales = (byte*)utmp;
            byte* mins = (byte*)(utmp + 2);
            int sumi = 0;
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
                sumi += y[i].Bsums[j] * mins[j / 2];

            int acc = 0;
            a = aux;
            sbyte* q8 = y[i].Qs;
            for (int j = 0; j < Qk.SuperBlock / 32; j++)
            {
                int scale = scales[j];
                for (int l = 0; l < 32; l++)
                    acc += scale * q8[l] * a[l];
                q8 += 32;
                a += 32;
            }

            float d = HalfBits.ToSingle(x[i].D) * y[i].D;
            float dmin = HalfBits.ToSingle(x[i].Dmin) * y[i].D;
            sumf += d * acc - dmin * sumi;
        }

        return sumf;
    }

    public static void Gemv(BlockQ5K* weights, float* input, float* output, int nIn, int nOut, CpuThreadPool? pool = null)
    {
        int nb = nIn / Qk.SuperBlock;
        using NativeBuffer q8 = new((nuint)Q8K.RowBytes(nIn));
        Q8K.QuantizeRow(input, (BlockQ8K*)q8.Pointer, nIn);
        BlockQ8K* y = (BlockQ8K*)q8.Pointer;
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

    public static void Gemm(BlockQ5K* weights, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null)
    {
        for (int t = 0; t < tokens; t++)
            Gemv(weights, input + t * nIn, output + t * nOut, nIn, nOut, pool);
    }
}
