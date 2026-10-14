namespace Flygon.Kernels;

/// <summary>One packed prefill weight (Q4 or Q6 panel) plus its output slab.</summary>
public unsafe struct PanelWeight
{
    public BlockQ4Kx8* Q4;
    public BlockQ4Kx8Meta* Meta;
    public BlockQ6Kx8* Q6;
    public float* Dst;
    public int NOut;

    public readonly bool IsEmpty => NOut == 0;

    public static PanelWeight ForQ4(BlockQ4Kx8* packed, BlockQ4Kx8Meta* meta, float* dst, int nOut) =>
        new() { Q4 = packed, Meta = meta, Dst = dst, NOut = nOut };

    public static PanelWeight ForQ6(BlockQ6Kx8* packed, float* dst, int nOut) =>
        new() { Q6 = packed, Dst = dst, NOut = nOut };
}

/// <summary>
/// Prefill fused ops: one parallel region quantizes src1 to q8_Kx4, spin-barriers, then
/// runs up to three panel GEMMs on the same hot q8 panel (llama.cpp keeps quantized
/// activations resident across back-to-back mul_mat the same way). Decode never enters here.
/// </summary>
public static unsafe class MulMatPanel
{
    public static void QuantizeAndGemm(float* input, BlockQ8Kx4* q8, int nIn, int tokens, CpuThreadPool? pool, PanelWeight w0, PanelWeight w1 = default, PanelWeight w2 = default)
    {
        int nb = nIn / Qk.SuperBlock;
        int quantGroups = tokens / 4;
        if (pool == null)
        {
            for (int g = 0; g < quantGroups; g++)
                QuantizeQ8Kx4.Quantize4x8(input + g * 4 * nIn, q8 + g * nb, nIn);
            GemmRange(w0, q8, nIn, tokens, 0, 1, nb);
            GemmRange(w1, q8, nIn, tokens, 0, 1, nb);
            GemmRange(w2, q8, nIn, tokens, 0, 1, nb);
            return;
        }

        int jobs = Math.Max(quantGroups, Math.Max(1, Math.Max(w0.NOut / 8, Math.Max(w1.NOut / 8, w2.NOut / 8))));
        pool.For(jobs, (int worker, int workers) =>
        {
            int g0 = quantGroups * worker / workers;
            int g1 = quantGroups * (worker + 1) / workers;
            for (int g = g0; g < g1; g++)
                QuantizeQ8Kx4.Quantize4x8(input + g * 4 * nIn, q8 + g * nb, nIn);
            pool.Barrier();
            GemmRange(w0, q8, nIn, tokens, worker, workers, nb);
            GemmRange(w1, q8, nIn, tokens, worker, workers, nb);
            GemmRange(w2, q8, nIn, tokens, worker, workers, nb);
        });
    }

    /// <summary>ffn_down: SiLU(gate)·up → q8_Kx4, barrier, one panel GEMM.</summary>
    public static void SiluQuantizeAndGemm(float* gate, float* up, BlockQ8Kx4* q8, int nIn, int tokens, CpuThreadPool? pool, PanelWeight w)
    {
        int nb = nIn / Qk.SuperBlock;
        int quantGroups = tokens / 4;
        if (pool == null)
        {
            for (int g = 0; g < quantGroups; g++)
                QuantizeQ8Kx4.Quantize4x8Silu(gate + g * 4 * nIn, up + g * 4 * nIn, q8 + g * nb, nIn);
            GemmRange(w, q8, nIn, tokens, 0, 1, nb);
            return;
        }

        int jobs = Math.Max(quantGroups, Math.Max(1, w.NOut / 8));
        pool.For(jobs, (int worker, int workers) =>
        {
            int g0 = quantGroups * worker / workers;
            int g1 = quantGroups * (worker + 1) / workers;
            for (int g = g0; g < g1; g++)
                QuantizeQ8Kx4.Quantize4x8Silu(gate + g * 4 * nIn, up + g * 4 * nIn, q8 + g * nb, nIn);
            pool.Barrier();
            GemmRange(w, q8, nIn, tokens, worker, workers, nb);
        });
    }

    private static void GemmRange(PanelWeight w, BlockQ8Kx4* q8, int nIn, int tokens, int worker, int workers, int nb)
    {
        int groups = w.NOut / 8;
        if (groups == 0)
            return;
        int begin = groups * worker / workers;
        int end = groups * (worker + 1) / workers;
        if (begin >= end)
            return;
        if (w.Q6 != null)
            GemmQ6K.Gemm8x8(nIn, w.Dst + begin * 8, w.NOut, w.Q6 + begin * nb, q8, tokens, (end - begin) * 8);
        else
            GemmQ4K.Gemm8x8(nIn, w.Dst + begin * 8, w.NOut, w.Q4 + begin * nb, q8, tokens, (end - begin) * 8, w.Meta + begin * nb);
    }
}
