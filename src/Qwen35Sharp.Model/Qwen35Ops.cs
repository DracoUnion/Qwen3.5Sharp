using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Flygon.Model;

/// <summary>
/// Additional pointwise ops needed by Qwen3.5: gated-RMSNorm (post-norm x
/// silu(gate)) and sigmoid helpers.
/// </summary>
public static unsafe class Qwen35Ops
{
    /// <summary>
    /// Gated RMSNorm: <c>y = (x * rsqrt(mean(x^2) + eps) * weight) * silu(gate)</c>.
    /// <paramref name="x"/> and <paramref name="y"/> are length <c>rows*dim</c>;
    /// <paramref name="gate"/> is same shape as x; <paramref name="weight"/> is
    /// length <c>dim</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GatedRmsNorm(
        float* x,
        float* weight,
        float* gate,
        float* y,
        int rows,
        int dim,
        float eps)
    {
        for (int r = 0; r < rows; r++)
        {
            float* xr = x + (long)r * dim;
            float* gr = gate + (long)r * dim;
            float* yr = y + (long)r * dim;
            float sum = 0f;
            int i = 0;
            if (Avx.IsSupported)
            {
                Vector256<float> acc = Vector256<float>.Zero;
                for (; i <= dim - 8; i += 8)
                {
                    Vector256<float> v = Avx.LoadVector256(xr + i);
                    acc = Fma.IsSupported ? Fma.MultiplyAdd(v, v, acc) : Avx.Add(acc, Avx.Multiply(v, v));
                }
                sum = Flygon.Kernels.VecDotQ4K.HorizontalSum(acc);
            }
            for (; i < dim; i++)
                sum += xr[i] * xr[i];
            float scale = 1f / MathF.Sqrt(sum / dim + eps);
            for (i = 0; i < dim; i++)
            {
                float w = weight != null ? weight[i] : 1f;
                float n = xr[i] * scale * w;
                float g = gr[i];
                yr[i] = n * (g / (1f + MathF.Exp(-g)));
            }
        }
    }

    /// <summary>sigmoid: y[i] = sigmoid(x[i]).</summary>
    public static void Sigmoid(float* x, float* y, int n)
    {
        for (int i = 0; i < n; i++)
            y[i] = 1f / (1f + MathF.Exp(-x[i]));
    }

    /// <summary>softplus(x) = log(1 + exp(x)), numerically stable.</summary>
    public static float SoftplusScalar(float x)
    {
        if (x > 20f)
            return x;
        // MathF has no Log1p; use the stable log(1+exp) identity.
        if (x < -10f)
            return MathF.Exp(x);
        float ex = MathF.Exp(x);
        return ex.Log1pSafe();
    }

    /// <summary>Safe log1p: for |x| &lt; 0.5 use the Taylor series; else log(1+x).</summary>
    internal static float Log1pSafe(this float x)
    {
        // log(1 + x); for x in (-1, inf). Use the series for |x| < 0.5.
        if (x > -0.5f && x < 0.5f)
        {
            float x2 = x * x;
            return x - x2 / 2f + x2 * x / 3f - x2 * x2 / 4f;
        }
        return MathF.Log(1f + x);
    }
}
