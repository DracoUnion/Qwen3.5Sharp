using System.Runtime.CompilerServices;

namespace Flygon.Kernels;

public static unsafe class Q4K
{
    public static void GetScaleMin(int j, byte* scales, out byte scale, out byte min)
    {
        if (j < 4)
        {
            scale = (byte)(scales[j] & 63);
            min = (byte)(scales[j + 4] & 63);
            return;
        }

        scale = (byte)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
        min = (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }

    public static void UnpackScales(byte* packed, uint* utmp)
    {
        const uint kmask1 = 0x3f3f3f3f;
        const uint kmask2 = 0x0f0f0f0f;
        const uint kmask3 = 0x03030303;
        utmp[0] = Unsafe.ReadUnaligned<uint>(packed);
        utmp[1] = Unsafe.ReadUnaligned<uint>(packed + 4);
        utmp[2] = Unsafe.ReadUnaligned<uint>(packed + 8);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        uint uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
    }

    public static void DequantizeRow(BlockQ4K* x, float* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            byte* q = x[i].Qs;
            float d = HalfBits.ToSingle(x[i].D);
            float min = HalfBits.ToSingle(x[i].Dmin);
            int iscale = 0;
            for (int j = 0; j < Qk.SuperBlock; j += 64)
            {
                GetScaleMin(iscale + 0, x[i].Scales, out byte sc0, out byte min0);
                GetScaleMin(iscale + 1, x[i].Scales, out byte sc1, out byte min1);
                float d1 = d * sc0;
                float m1 = min * min0;
                float d2 = d * sc1;
                float m2 = min * min1;
                for (int l = 0; l < 32; l++)
                    *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; l++)
                    *y++ = d2 * (q[l] >> 4) - m2;
                q += 32;
                iscale += 2;
            }
        }
    }

    /// <summary>
    /// Valid Q4_K pack for tests and leftovers. Not the ggml importance-weighted
    /// quantizer — GGUF weights already arrive quantized.
    /// </summary>
    public static void PackSimple(float* x, BlockQ4K* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            float* row = x + i * Qk.SuperBlock;
            float[] mins = new float[8];
            float[] scales = new float[8];
            byte[] levels = new byte[Qk.SuperBlock];
            for (int g = 0; g < 8; g++)
            {
                float lo = row[g * 32];
                float hi = lo;
                for (int t = 1; t < 32; t++)
                {
                    float v = row[g * 32 + t];
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }

                mins[g] = -lo;
                float range = hi - lo;
                scales[g] = range > 0 ? range / 15f : 0f;
                if (scales[g] == 0)
                {
                    for (int t = 0; t < 32; t++)
                        levels[g * 32 + t] = 0;
                    continue;
                }

                for (int t = 0; t < 32; t++)
                {
                    int q = (int)MathF.Round((row[g * 32 + t] - lo) / scales[g]);
                    if (q < 0) q = 0;
                    if (q > 15) q = 15;
                    levels[g * 32 + t] = (byte)q;
                }
            }

            float maxScale = 0;
            float maxMin = 0;
            for (int g = 0; g < 8; g++)
            {
                if (scales[g] > maxScale) maxScale = scales[g];
                if (mins[g] > maxMin) maxMin = mins[g];
            }

            float invScale = maxScale > 0 ? 63f / maxScale : 0f;
            float invMin = maxMin > 0 ? 63f / maxMin : 0f;
            byte* packed = y[i].Scales;
            for (int t = 0; t < 12; t++)
                packed[t] = 0;
            for (int g = 0; g < 8; g++)
            {
                int ls = (int)MathF.Round(invScale * scales[g]);
                int lm = (int)MathF.Round(invMin * mins[g]);
                if (ls > 63) ls = 63;
                if (lm > 63) lm = 63;
                if (g < 4)
                {
                    packed[g] = (byte)ls;
                    packed[g + 4] = (byte)lm;
                }
                else
                {
                    packed[g + 4] = (byte)((ls & 0xF) | ((lm & 0xF) << 4));
                    packed[g - 4] |= (byte)((ls >> 4) << 6);
                    packed[g] |= (byte)((lm >> 4) << 6);
                }
            }

            y[i].D = HalfBits.FromSingle(maxScale / 63f);
            y[i].Dmin = HalfBits.FromSingle(maxMin / 63f);

            for (int g = 0; g < 8; g++)
            {
                GetScaleMin(g, packed, out byte sc, out byte m);
                float d = HalfBits.ToSingle(y[i].D) * sc;
                if (d == 0)
                    continue;
                float dm = HalfBits.ToSingle(y[i].Dmin) * m;
                for (int t = 0; t < 32; t++)
                {
                    int q = (int)MathF.Round((row[g * 32 + t] + dm) / d);
                    if (q < 0) q = 0;
                    if (q > 15) q = 15;
                    levels[g * 32 + t] = (byte)q;
                }
            }

            byte* qs = y[i].Qs;
            for (int j = 0; j < Qk.SuperBlock; j += 64)
            {
                for (int l = 0; l < 32; l++)
                    qs[l] = (byte)(levels[j + l] | (levels[j + l + 32] << 4));
                qs += 32;
            }
        }
    }
}
