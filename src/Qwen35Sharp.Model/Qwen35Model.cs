using System.Diagnostics;
using System.Runtime.InteropServices;
using Flygon.Gguf;
using Flygon.Kernels;

namespace Flygon.Model;

/// <summary>
/// Qwen3.5 hybrid transformer: Gated-DeltaNet linear-attention layers
/// interleaved with full-attention layers. KV cache for the full-attention
/// layers; per-layer conv/recurrent state for the linear-attention layers.
/// </summary>
public sealed unsafe partial class Qwen35Model : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly CpuThreadPool _pool;
    private readonly ScratchArena _gemmScratch = new();
    private readonly Dictionary<string, Weight> _weights = new(StringComparer.Ordinal);
    private readonly List<NativeBuffer> _buffers = [];
    private NativeBuffer? _scratch;
    private int _scratchUsed;

    // KV cache (full-attention layers only).
    private readonly float*[] _cacheK;
    private readonly float*[] _cacheV;
    private readonly NativeBuffer[] _cacheKBuffers;
    private readonly NativeBuffer[] _cacheVBuffers;
    private readonly List<int> _linearLayers = [];
    private readonly List<int> _fullLayers = [];

    // Recurrent state for linear-attention layers: [nHeads][headDim][vHeadDim]
    private float*[] _linState;
    // Conv state for linear-attention layers: [convKernelSize - 1][convDim]
    private float*[] _linConv;
    private NativeBuffer? _linStateBuffer;
    private NativeBuffer? _linConvBuffer;

    public ModelConfig Config { get; private set; }
    public BpeTokenizer Tokenizer { get; }
    public int ThreadCount => _pool.ThreadCount;
    public string ThreadAutoHint => CpuTopology.AutoHint;
    public static bool ProfileEnabled;
    public static long TicksQ4;
    public static long TicksQ2;
    public static long TicksSTQ;
    public static long TicksQ6;
    public static long TicksAttnScore;
    public static long TicksSoftmax;
    public static long TicksAttnCombine;
    public static long TicksRms;
    public static long TicksRope;
    public static long TicksSilu;
    public static long TicksEmbed;
    public static long TicksDelta;
    public static long TicksConv;

    public static void ResetProfile()
    {
        TicksQ4 = 0;
        TicksQ2 = 0;
        TicksSTQ = 0;
        TicksQ6 = 0;
        TicksAttnScore = 0;
        TicksSoftmax = 0;
        TicksAttnCombine = 0;
        TicksRms = 0;
        TicksRope = 0;
        TicksSilu = 0;
        TicksEmbed = 0;
        TicksDelta = 0;
        TicksConv = 0;
    }

    private readonly List<int> _cacheTokens = [];
    private int _cacheLen;
    private int _cacheCap;

    public Qwen35Model(string ggufPath, int threads = 0)
    {
        _gguf = new GgufFile(ggufPath);
        Config = ModelConfig.FromGguf(_gguf);
        if (!string.Equals(Config.Architecture, "qwen35", StringComparison.Ordinal))
            throw new NotSupportedException($"Expected qwen35, got {Config.Architecture}");

        Tokenizer = new BpeTokenizer(_gguf);
        _pool = new CpuThreadPool(threads);
        LoadWeights();
        if (Config.VocabSize == 0 && _weights.TryGetValue("token_embd.weight", out Weight emb) && emb.NOut > 0)
            Config.VocabSize = emb.NOut;

        // Index the two layer families.
        for (int l = 0; l < Config.NumLayers; l++)
        {
            if (Config.IsRecurrent[l])
                _linearLayers.Add(l);
            else
                _fullLayers.Add(l);
        }

        // Allocate per-layer state for linear layers (recurrent + conv state).
        int linHeads = Config.SsmTimeStepRank;
        int linHeadDim = Config.SsmStateSize;
        int linVHeadDim = Config.SsmStateSize;
        _linState = new float*[Config.NumLayers];
        _linConv = new float*[Config.NumLayers];
        long stateBytes = (long)linHeads * linHeadDim * linVHeadDim * sizeof(float);
        int stateLen = Math.Max(1, Config.SsmConvKernel - 1);
        long convBytes = (long)stateLen * Config.LinearConvDim * sizeof(float);
        NativeBuffer sb = Rent((nuint)((long)Config.NumLayers * stateBytes));
        NativeBuffer cb = Rent((nuint)((long)Config.NumLayers * convBytes));
        _linStateBuffer = sb;
        _linConvBuffer = cb;
        for (int l = 0; l < Config.NumLayers; l++)
        {
            _linState[l] = (float*)((byte*)sb.Pointer + (long)l * stateBytes);
            _linConv[l] = (float*)((byte*)cb.Pointer + (long)l * convBytes);
        }

        // KV cache only for full-attention layers.
        _cacheCap = Math.Min(Config.ContextLength, 4096);
        _cacheK = new float*[Config.NumLayers];
        _cacheV = new float*[Config.NumLayers];
        _cacheKBuffers = new NativeBuffer[Config.NumLayers];
        _cacheVBuffers = new NativeBuffer[Config.NumLayers];
        int kvStride = Config.FullKvDim;
        for (int l = 0; l < Config.NumLayers; l++)
        {
            if (Config.IsRecurrent[l])
                continue;
            NativeBuffer k = Rent((nuint)((long)_cacheCap * kvStride * sizeof(float)));
            NativeBuffer v = Rent((nuint)((long)_cacheCap * kvStride * sizeof(float)));
            _cacheKBuffers[l] = k;
            _cacheVBuffers[l] = v;
            _cacheK[l] = (float*)k.Pointer;
            _cacheV[l] = (float*)v.Pointer;
        }

        // Scratch is sized lazily per Forward call from the actual sequence length.
        // This avoids reserving gigabytes for the configured context window.
    }

    public int CacheLength => _cacheLen;

    public void ResetCache()
    {
        TruncateCache(0);
        // Clear all per-layer recurrent / conv state.
        for (int l = 0; l < Config.NumLayers; l++)
        {
            if (Config.IsRecurrent[l])
            {
                NativeMemory.Clear(_linState[l], (nuint)((long)Config.SsmTimeStepRank * Config.SsmStateSize * Config.SsmStateSize * sizeof(float)));
                NativeMemory.Clear(_linConv[l], (nuint)((long)Math.Max(1, Config.SsmConvKernel - 1) * Config.LinearConvDim * sizeof(float)));
            }
        }
    }

    public void TruncateCache(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (length > _cacheLen)
            throw new ArgumentOutOfRangeException(nameof(length), length, $"Cannot extend cache from {_cacheLen}.");
        _cacheLen = length;
        if (_cacheTokens.Count > length)
            _cacheTokens.RemoveRange(length, _cacheTokens.Count - length);
    }

    public PromptAlignment AlignPrompt(ReadOnlySpan<int> promptIds)
    {
        PromptReuse plan = KvCacheAlign.Plan(CollectionsMarshal.AsSpan(_cacheTokens), promptIds);
        TruncateCache(plan.TruncateTo);
        return new PromptAlignment(plan.TruncateTo, promptIds[plan.SuffixStart..].ToArray());
    }

    public float[] Forward(int[] tokens)
    {
        _scratchUsed = 0;
        int seq = tokens.Length;
        int start = _cacheLen;
        EnsureCache(start + seq);
        EnsureScratch(seq, start + seq);
        int hidden = Config.HiddenSize;
        float* h = (float*)Bump((nuint)((long)seq * hidden * sizeof(float)));
        Embed(tokens, h);
        int layerMark = _scratchUsed;
        if (seq == 1)
        {
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                _scratchUsed = layerMark;
                DecodeBlock(h, layer, start);
            }
        }
        else
        {
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                _scratchUsed = layerMark;
                PrefillBlock(h, layer, seq, start);
            }
        }

        float* normed = (float*)Bump((nuint)((long)seq * hidden * sizeof(float)));
        Rms("output_norm.weight", h, normed, seq, hidden);
        float* last = normed + (seq - 1) * hidden;

        int vocab = Config.VocabSize > 0 ? Config.VocabSize : Tokenizer.VocabSize;
        float[] logits = new float[vocab];
        fixed (float* lp = logits)
            Linear(last, "output.weight", lp, hidden, vocab, 1, fallback: "token_embd.weight");

        _cacheLen += seq;
        _cacheTokens.AddRange(tokens);
        Debug.Assert(_cacheTokens.Count == _cacheLen);
        return logits;
    }

    private void EnsureCache(int needed)
    {
        if (needed <= _cacheCap)
            return;
        int next = Math.Max(needed, _cacheCap * 2);
        next = Math.Min(next, Config.ContextLength);
        int kvStride = Config.FullKvDim;
        NativeBuffer?[] nextK = new NativeBuffer?[Config.NumLayers];
        NativeBuffer?[] nextV = new NativeBuffer?[Config.NumLayers];
        try
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (Config.IsRecurrent[l])
                    continue;
                NativeBuffer k = Rent((nuint)((long)next * kvStride * sizeof(float)));
                NativeBuffer v = Rent((nuint)((long)next * kvStride * sizeof(float)));
                nextK[l] = k;
                nextV[l] = v;
                long copyBytes = (long)_cacheLen * kvStride * sizeof(float);
                Buffer.MemoryCopy(_cacheK[l], k.Pointer, copyBytes, copyBytes);
                Buffer.MemoryCopy(_cacheV[l], v.Pointer, copyBytes, copyBytes);
            }
        }
        catch
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (nextK[l] is NativeBuffer k)
                {
                    _buffers.Remove(k);
                    k.Dispose();
                }
                if (nextV[l] is NativeBuffer v)
                {
                    _buffers.Remove(v);
                    v.Dispose();
                }
            }
            throw;
        }

        for (int l = 0; l < Config.NumLayers; l++)
        {
            if (Config.IsRecurrent[l])
                continue;
            NativeBuffer oldK = _cacheKBuffers[l];
            NativeBuffer oldV = _cacheVBuffers[l];
            NativeBuffer k = nextK[l]!;
            NativeBuffer v = nextV[l]!;
            _cacheKBuffers[l] = k;
            _cacheVBuffers[l] = v;
            _cacheK[l] = (float*)k.Pointer;
            _cacheV[l] = (float*)v.Pointer;
            _buffers.Remove(oldK);
            _buffers.Remove(oldV);
            oldK.Dispose();
            oldV.Dispose();
        }
        _cacheCap = next;
    }

    private NativeBuffer Rent(nuint bytes)
    {
        NativeBuffer buf = new(bytes);
        _buffers.Add(buf);
        return buf;
    }

    /// <summary>
    /// Grow the per-layer scratch arena so a single layer forward never regrows
    /// mid-call (which would invalidate already-issued pointers). Sizes from the
    /// actual tokens processed this call and the current KV length, not the
    /// configured context window.
    /// </summary>
    private void EnsureScratch(int seq, int kvLen)
    {
        int hidden = Config.HiddenSize;
        int ffn = Config.FfnSize;
        int nHeads = Config.SsmTimeStepRank;
        int vHeadDim = Config.SsmStateSize;
        int convDim = Config.LinearConvDim;
        int keyDim = Config.LinearKeyDim;
        int valueDim = Config.LinearValueDim;
        int heads = Config.NumHeads;
        int qDim = Config.FullQDim;

        long linInternal =
            (long)seq * convDim * sizeof(float)              // qkv
            + (long)seq * valueDim * sizeof(float)           // z
            + (long)seq * convDim * 2 * sizeof(float)        // qkvCh + convOutCh
            + (long)seq * (keyDim + keyDim + valueDim) * sizeof(float) // qSm/kSm/vSm
            + (long)seq * nHeads * 4 * sizeof(float)         // bOut, aOut, betaP, gP
            + (long)seq * nHeads * vHeadDim * 2 * sizeof(float) // out_, gated
            + (long)cached_gather_norm(hidden);              // 2 norms + attn (outside, counted separately)
        long fullInternal =
            (long)seq * 2 * qDim * sizeof(float)             // qGateBuf
            + (long)seq * qDim * 2 * sizeof(float)           // q + gate
            + (long)heads * seq * kvLen * sizeof(float)       // scores
            + (long)seq * qDim * sizeof(float);              // ao
        long ffnScratch = (long)seq * (ffn + ffn + hidden) * sizeof(float);
        long normScratch = (long)seq * hidden * 3 * sizeof(float);
        long needed = Math.Max(linInternal, fullInternal) + ffnScratch + normScratch + 4 * 1024 * 1024;
        if (_scratch == null || (long)_scratch.Bytes < needed)
            _scratch = Rent((nuint)needed);
    }

    // Small helper to document the extra norm buffers allocated by DecodeBlock.
    private static long cached_gather_norm(int hidden) => (long)hidden * 4 * sizeof(float);

    private void* Bump(nuint bytes)
    {
        // EnsureScratch is called once at the start of each Forward and sizes the
        // arena for the whole pass, so a regrow mid-call must never happen (it
        // would invalidate pointers already handed out to the current layer).
        if (_scratch == null)
            _scratch = Rent(64 * 1024 * 1024);
        int aligned = (int)((bytes + 63) & ~(nuint)63);
        if (_scratchUsed + aligned > (int)_scratch.Bytes)
            throw new InsufficientMemoryException(
                $"Scratch arena overflow: need {_scratchUsed + aligned} of {_scratch.Bytes} bytes. " +
                "Recompute EnsureScratch sizing for this model.");
        byte* ptr = (byte*)_scratch.Pointer + _scratchUsed;
        _scratchUsed += aligned;
        return ptr;
    }

    public void Dispose()
    {
        foreach (NativeBuffer buf in _buffers)
            buf.Dispose();
        _gemmScratch.Dispose();
        _pool.Dispose();
        _gguf.Dispose();
    }

    // Weight is a struct that packs the raw pointer for whatever quant format
    // a tensor happens to have; the dispatch helpers switch on Type.
    private struct Weight
    {
        public GgmlTensorType Type;
        public BlockQ4K* Q4;
        public BlockQ4Kx8* Packed;
        public BlockQ4Kx8Meta* Meta;
        public BlockQ5K* Q5;
        public BlockQ6K* Q6;
        public BlockQ2_0C* Q2;
        public BlockQ2x8* Packed2;
        public BlockSTQ1_0* STQ;
        public BlockSTQ1_0x8* PackedSTQ;
        public BlockQ6Kx8* Packed6;
        public float* F32;
        public int NIn;
        public int NOut;
        public int Count;
    }

    private void LoadWeights()
    {
        foreach ((string name, GgufTensorInfo info) in _gguf.Tensors)
        {
            byte* data = _gguf.DataBase + (long)info.Offset;
            if (info.Type == GgmlTensorType.F32)
            {
                int n = checked((int)info.NumElements);
                NativeBuffer buf = Rent((nuint)((long)n * sizeof(float)));
                Buffer.MemoryCopy(data, buf.Pointer, buf.Bytes, buf.Bytes);
                _weights[name] = new Weight { Type = GgmlTensorType.F32, F32 = (float*)buf.Pointer, Count = n };
                continue;
            }

            int nIn = (int)info.Shape[0];
            int nOut = info.Shape.Length > 1 ? (int)info.Shape[1] : 1;
            if (nIn % Qk.SuperBlock != 0)
                throw new NotSupportedException($"{name}: nIn={nIn} is not a multiple of 256");
            int nb = info.Type switch
            {
                GgmlTensorType.Q2_0C => nIn / Q2_0C.BlockLength,
                GgmlTensorType.STQ1_0 => nIn / STQ1_0.BlockLength,
                _ => nIn / Qk.SuperBlock,
            };
            int blockBytes = info.Type switch
            {
                GgmlTensorType.Q4_K => Qk.Q4KSize,
                GgmlTensorType.Q5_K => Qk.Q5KSize,
                GgmlTensorType.Q6_K => Qk.Q6KSize,
                GgmlTensorType.Q2_0C => Qk.Q2_0CSize,
                GgmlTensorType.STQ1_0 => Qk.STQ1_0Size,
                _ => throw new NotSupportedException($"Unsupported quantized tensor type {info.Type} for {name}"),
            };
            NativeBuffer quant = Rent((nuint)((long)nOut * nb * blockBytes));
            Buffer.MemoryCopy(data, quant.Pointer, quant.Bytes, quant.Bytes);
            Weight w = new()
            {
                Type = info.Type,
                NIn = nIn,
                NOut = nOut,
            };
            if (info.Type == GgmlTensorType.Q4_K)
            {
                NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q4Kx8Size));
                if (nOut >= 8)
                {
                    RepackQ4K.Rows((BlockQ4K*)quant.Pointer, (BlockQ4Kx8*)packed.Pointer, nIn, nOut & ~7);
                    NativeBuffer meta = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q4Kx8MetaSize));
                    RepackQ4K.BuildMeta((BlockQ4K*)quant.Pointer, (BlockQ4Kx8Meta*)meta.Pointer, nIn, nOut & ~7);
                    w.Meta = (BlockQ4Kx8Meta*)meta.Pointer;
                }
                w.Q4 = (BlockQ4K*)quant.Pointer;
                w.Packed = (BlockQ4Kx8*)packed.Pointer;
            }
            else if (info.Type == GgmlTensorType.Q5_K)
            {
                w.Q5 = (BlockQ5K*)quant.Pointer;
            }
            else
            {
                if (info.Type == GgmlTensorType.Q2_0C)
                {
                    w.Q2 = (BlockQ2_0C*)quant.Pointer;
                    if (nOut >= 8)
                    {
                        NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q2x8Size));
                        RepackQ2.Rows(w.Q2, (BlockQ2x8*)packed.Pointer, nIn, nOut & ~7);
                        w.Packed2 = (BlockQ2x8*)packed.Pointer;
                    }
                }
                else if (info.Type == GgmlTensorType.STQ1_0)
                {
                    w.STQ = (BlockSTQ1_0*)quant.Pointer;
                    if (nOut >= 8)
                    {
                        NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.STQ1_0x8Size));
                        RepackSTQ.Rows(w.STQ, (BlockSTQ1_0x8*)packed.Pointer, nIn, nOut & ~7);
                        w.PackedSTQ = (BlockSTQ1_0x8*)packed.Pointer;
                    }
                }
                else
                {
                    w.Q6 = (BlockQ6K*)quant.Pointer;
                    if (nOut >= 8)
                    {
                        NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q6Kx8Size));
                        RepackQ6K.Rows(w.Q6, (BlockQ6Kx8*)packed.Pointer, nIn, nOut & ~7);
                        w.Packed6 = (BlockQ6Kx8*)packed.Pointer;
                    }
                }
            }

            _weights[name] = w;
        }
    }
}
