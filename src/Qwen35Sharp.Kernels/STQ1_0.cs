using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Flygon.Kernels;

/// <summary>
/// Sherry/STQ1_0 sparse ternary dot products (canonical GGUF type 43;
/// legacy HyMT2 files are normalized from raw type 42 by GgufFile).
/// Each 256-weight block stores 64 stride-16 groups.  Every group has one
/// zero and three values of equal magnitude, encoded by a 32-entry codebook.
/// </summary>
public static unsafe class STQ1_0
{
    public const int BlockLength = Qk.STQ1_0BlockLength;

    private static readonly Vector128<byte> CodebookLo = Vector128.Create(
        (byte)0xA9, (byte)0x89, (byte)0x29, (byte)0x09,
        (byte)0xA6, (byte)0x86, (byte)0x26, (byte)0x06,
        (byte)0x9A, (byte)0x92, (byte)0x1A, (byte)0x12,
        (byte)0x6A, (byte)0x62, (byte)0x4A, (byte)0x42);
    private static readonly Vector128<byte> CodebookHi = Vector128.Create(
        (byte)0x01, (byte)0x21, (byte)0x81, (byte)0xA1,
        (byte)0x04, (byte)0x24, (byte)0x84, (byte)0xA4,
        (byte)0x10, (byte)0x18, (byte)0x90, (byte)0x98,
        (byte)0x40, (byte)0x48, (byte)0x60, (byte)0x68);
    private static readonly Vector128<byte> NibbleMask = Vector128.Create((byte)0x0F);
    private static readonly Vector128<byte> LaneMask = Vector128.Create((byte)0x03);
    private static readonly Vector128<short> Ones = Vector128.Create((short)1);
    // Eight 0/0xff sign selectors per byte.  Keeping the expansion in a
    // vector LUT removes the stackalloc and bit loop from every 16-group
    // chunk in the AVX2 fallback/tail path.
    private static readonly Vector128<byte>[] SignMaskLut = BuildSignMaskLut();

    private static Vector128<byte>[] BuildSignMaskLut()
    {
        Vector128<byte>[] table = new Vector128<byte>[256];
        for (int value = 0; value < table.Length; value++)
        {
            ulong bits = 0;
            for (int i = 0; i < 8; i++)
                if (((value >> i) & 1) != 0)
                    bits |= 0xFFUL << (i * 8);
            table[value] = Vector128.CreateScalar(bits).AsByte();
        }
        return table;
    }

    // Packed four-lane ternary patterns.  A lane uses 2 bits: 0=-1, 1=0,
    // 2=+1.  The index is (sign << 4) | slot.
    private static ReadOnlySpan<byte> Codebook =>
    [
        0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
        0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
        0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
        0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
    ];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static byte QPack(BlockSTQ1_0* x, int group)
    {
        int slot = (x->Qs[group >> 1] >> ((group & 1) * 4)) & 0x0F;
        int sign = (x->Sign[group >> 3] >> (group & 7)) & 1;
        return Codebook[(sign << 4) | slot];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int Lane(byte qpack, int lane) => ((qpack >> (lane * 2)) & 3) - 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static byte Code(BlockSTQ1_0* x, int index)
    {
        int chunk = index >> 6;
        int gloc = index & 15;
        int lane = (index >> 4) & 3;
        return (byte)((QPack(x, (chunk << 4) + gloc) >> (lane * 2)) & 3);
    }

    public static void DequantizeRow(BlockSTQ1_0* x, float* dst, int n)
    {
        if (n <= 0 || n % BlockLength != 0)
            throw new ArgumentException("STQ1_0 rows must contain a positive multiple of 256 values.", nameof(n));

        for (int b = 0; b < n / BlockLength; b++)
        {
            float d = HalfBits.ToSingle(x[b].D);
            for (int g = 0; g < BlockLength / 4; g++)
            {
                byte qpack = QPack(x + b, g);
                int chunk = g >> 4;
                int gloc = g & 15;
                for (int lane = 0; lane < 4; lane++)
                    dst[b * BlockLength + chunk * 64 + gloc + lane * 16] = Lane(qpack, lane) * d;
            }
        }
    }

    public static float Dot(BlockSTQ1_0* x, BlockQ8K* y, int n) =>
        Avx2.IsSupported ? DotAvx2(x, y, n) : DotScalar(x, y, n);

    public static float DotScalar(BlockSTQ1_0* x, BlockQ8K* y, int n)
    {
        if (n <= 0 || n % BlockLength != 0)
            throw new ArgumentException("STQ1_0 rows must contain a positive multiple of 256 values.", nameof(n));

        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            int si = 0;
            for (int g = 0; g < BlockLength / 4; g++)
            {
                byte qpack = QPack(x + b, g);
                int chunk = g >> 4;
                int gloc = g & 15;
                for (int lane = 0; lane < 4; lane++)
                    si += Lane(qpack, lane) * y[b].Qs[chunk * 64 + gloc + lane * 16];
            }
            sum += HalfBits.ToSingle(x[b].D) * y[b].D * si;
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotAvx2(BlockSTQ1_0* x, BlockQ8K* y, int n)
    {
        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            int encoded = 0;
            int activationSum = 0;
            for (int chunk = 0; chunk < 4; chunk++)
            {
                encoded += DotChunk16(x + b, y + b, chunk);
                for (int i = 0; i < 4; i++)
                    activationSum += y[b].Bsums[chunk * 4 + i];
            }

            sum += HalfBits.ToSingle(x[b].D) * y[b].D * (encoded - activationSum);
        }
        return sum;
    }

    /// <summary>
    /// Computes the encoded 0/1/2 dot for one 64-value stride-16 chunk (16
    /// groups).  The caller subtracts the Q8 activation sum to turn q=0/1/2
    /// into the signed ternary values -1/0/+1.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int DotChunk16(BlockSTQ1_0* x, BlockQ8K* y, int chunk)
    {
        byte* qs = x->Qs + chunk * 8;
        byte* signs = x->Sign + chunk * 2;
        sbyte* activations = y->Qs + chunk * 64;

        // Eight bytes contain sixteen 4-bit codebook slots.  Interleaving the
        // low and high nibbles gives one slot per byte for vpshufb.
        ulong packed = Unsafe.ReadUnaligned<ulong>(qs);
        Vector128<byte> qbytes = Vector128.CreateScalar(packed).AsByte();
        Vector128<byte> lo = Sse2.And(qbytes, NibbleMask);
        Vector128<byte> hi = Sse2.And(Sse2.ShiftRightLogical(qbytes.AsUInt16(), 4).AsByte(), NibbleMask);
        Vector128<byte> slots = Sse2.UnpackLow(lo, hi);

        Vector128<byte> signMask = Sse2.Or(
            SignMaskLut[signs[0]],
            Sse2.ShiftLeftLogical128BitLane(SignMaskLut[signs[1]], 8));

        Vector128<byte> q0 = Ssse3.Shuffle(CodebookLo, slots);
        Vector128<byte> q1 = Ssse3.Shuffle(CodebookHi, slots);
        Vector128<byte> qpack = Sse2.Or(Sse2.AndNot(signMask, q0), Sse2.And(signMask, q1));

        Vector128<int> acc = Vector128<int>.Zero;
        Vector128<byte> q = Sse2.And(qpack, LaneMask);
        Vector128<sbyte> a = Sse2.LoadVector128(activations);
        Vector128<short> pair = Ssse3.MultiplyAddAdjacent(q, a);
        acc = Sse2.Add(acc, Sse2.MultiplyAddAdjacent(pair, Ones));
        q = Sse2.And(Sse2.ShiftRightLogical(qpack.AsUInt16(), 2).AsByte(), LaneMask);
        a = Sse2.LoadVector128(activations + 16);
        pair = Ssse3.MultiplyAddAdjacent(q, a);
        acc = Sse2.Add(acc, Sse2.MultiplyAddAdjacent(pair, Ones));
        q = Sse2.And(Sse2.ShiftRightLogical(qpack.AsUInt16(), 4).AsByte(), LaneMask);
        a = Sse2.LoadVector128(activations + 32);
        pair = Ssse3.MultiplyAddAdjacent(q, a);
        acc = Sse2.Add(acc, Sse2.MultiplyAddAdjacent(pair, Ones));
        q = Sse2.And(Sse2.ShiftRightLogical(qpack.AsUInt16(), 6).AsByte(), LaneMask);
        a = Sse2.LoadVector128(activations + 48);
        pair = Ssse3.MultiplyAddAdjacent(q, a);
        acc = Sse2.Add(acc, Sse2.MultiplyAddAdjacent(pair, Ones));

        acc = Ssse3.HorizontalAdd(acc, acc);
        acc = Ssse3.HorizontalAdd(acc, acc);
        return acc.ToScalar();
    }
}
