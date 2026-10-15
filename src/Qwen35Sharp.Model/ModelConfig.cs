using Flygon.Gguf;

namespace Flygon.Model;

/// <summary>
/// Qwen3.5 is a hybrid transformer: some layers use gated-DeltaNet linear
/// attention, the rest use standard full attention. The FFN is either a dense
/// SiLU MLP (arch <c>qwen35</c>) or a sparse MoE block (arch
/// <c>qwen35moe</c>). This config captures the dimension parameters, which
/// layers are linear, and whether the FFN is MoE.
/// </summary>
public sealed class ModelConfig
{
    public string Architecture { get; init; } = "qwen35";
    public int NumLayers { get; init; }
    public int HiddenSize { get; init; }
    public int FfnSize { get; init; }

    // Full-attention parameters.
    public int NumHeads { get; init; }
    public int NumKvHeads { get; init; }
    public int HeadDim { get; init; }
    public int RopeDim { get; init; }
    public float RopeBase { get; init; } = 10000f;

    // Gated-DeltaNet (linear attention) parameters.
    public int SsmConvKernel { get; init; }      // ssm.conv_kernel
    public int SsmInnerSize { get; init; }        // ssm.inner_size (== value dim)
    public int SsmStateSize { get; init; }        // ssm.state_size (== key head dim)
    public int SsmTimeStepRank { get; init; }     // ssm.time_step_rank (== number of value heads)
    public int SsmGroupCount { get; init; }        // ssm.group_count (== number of key heads)

    // MoE FFN parameters (only for arch "qwen35moe").
    public bool IsMoe { get; init; }
    public int NumExperts { get; init; }            // expert_count
    public int NumExpertsPerTok { get; init; }      // expert_used_count (top-k)
    public int MoeFfnSize { get; init; }            // expert_feed_forward_length (per-expert FFN dim)
    public int MoeSharedFfnSize { get; init; }      // expert_shared_feed_forward_length
    public int MoeSharedCount { get; init; }        // expert_shared_count (>= 1 when shared experts present)
    public float ExpertWeightsScale { get; init; }  // expert_weights_scale (> 0 scales expert outputs)

    public int VocabSize { get; set; }
    public int ContextLength { get; init; }
    public float Eps { get; init; } = 1e-6f;

    /// <summary>Per-layer type: true = linear attention (Gated-DeltaNet), false = full attention.</summary>
    public bool[] IsRecurrent { get; init; } = [];

    public int LinearConvDim => 2 * SsmGroupCount * SsmStateSize + SsmInnerSize;
    public int LinearKeyDim => SsmGroupCount * SsmStateSize;
    public int LinearValueDim => SsmTimeStepRank * SsmStateSize;
    public int LinearQkvDim => 2 * SsmStateSize * SsmGroupCount + SsmInnerSize;
    public int FullQDim => NumHeads * HeadDim;
    public int FullKvDim => NumKvHeads * HeadDim;

    public static ModelConfig FromGguf(GgufFile gguf)
    {
        string arch = gguf.GetString("general.architecture", "qwen35");
        int hidden = (int)gguf.GetUint32($"{arch}.embedding_length");
        int heads = (int)gguf.GetUint32($"{arch}.attention.head_count");
        int kvHeads = (int)gguf.GetUint32($"{arch}.attention.head_count_kv", (uint)heads);
        int headDim = (int)gguf.GetUint32($"{arch}.attention.key_length", 0);
        if (headDim == 0)
            headDim = hidden / heads;
        int ropeDim = (int)gguf.GetUint32($"{arch}.rope.dimension_count", (uint)headDim);
        float ropeBase = gguf.GetFloat32($"{arch}.rope.freq_base", 10000f);

        // Gated-DeltaNet parameters.
        int convKernel = (int)gguf.GetUint32($"{arch}.ssm.conv_kernel", 4u);
        int innerSize = (int)gguf.GetUint32($"{arch}.ssm.inner_size");
        int stateSize = (int)gguf.GetUint32($"{arch}.ssm.state_size", (uint)headDim);
        int timeStepRank = (int)gguf.GetUint32($"{arch}.ssm.time_step_rank");
        int groupCount = (int)gguf.GetUint32($"{arch}.ssm.group_count");

        // Layer type is decided by which attention tensors are present:
        // a linear (Gated-DeltaNet) layer has blk.L.attn_qkv.weight; a full
        // attention layer has blk.L.attn_q.weight. Falling back to the
        // full_attention_interval formula covers checkpoints that follow it.
        int layers = (int)gguf.GetUint32($"{arch}.block_count");
        int fullAttnInterval = gguf.GetInt32($"{arch}.full_attention_interval", 4);
        bool[] isRecurrent = new bool[layers];
        bool anyTensors = false;
        for (int i = 0; i < layers; i++)
        {
            bool hasQkv = gguf.Tensors.ContainsKey($"blk.{i}.attn_qkv.weight");
            bool hasQ = gguf.Tensors.ContainsKey($"blk.{i}.attn_q.weight");
            if (hasQkv || hasQ)
            {
                isRecurrent[i] = hasQkv;
                anyTensors = true;
            }
            else
            {
                isRecurrent[i] = (i + 1) % fullAttnInterval != 0;
            }
        }
        // If no layer tensors were present (header-only config?), use the formula
        // for all layers.
        if (!anyTensors)
        {
            for (int i = 0; i < layers; i++)
                isRecurrent[i] = (i + 1) % fullAttnInterval != 0;
        }

        // MoE branch: read expert params if the arch is qwen35moe.
        bool isMoe = string.Equals(arch, "qwen35moe", StringComparison.Ordinal);
        int numExperts = (int)gguf.GetUint32($"{arch}.expert_count", 0);
        int usedCount = (int)gguf.GetUint32($"{arch}.expert_used_count", isMoe ? 8u : 0u);
        int expFfn = (int)gguf.GetUint32($"{arch}.expert_feed_forward_length", 0);
        int sharedFfn = (int)gguf.GetUint32($"{arch}.expert_shared_feed_forward_length", 0);
        int sharedCount = (int)gguf.GetUint32($"{arch}.expert_shared_count", 0);
        float expScale = gguf.GetFloat32($"{arch}.expert_weights_scale", 0f);

        return new ModelConfig
        {
            Architecture = arch,
            NumLayers = layers,
            HiddenSize = hidden,
            FfnSize = (int)gguf.GetUint32($"{arch}.feed_forward_length", 0u),
            NumHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = headDim,
            RopeDim = ropeDim,
            RopeBase = ropeBase,
            SsmConvKernel = convKernel,
            SsmInnerSize = innerSize,
            SsmStateSize = stateSize,
            SsmTimeStepRank = timeStepRank,
            SsmGroupCount = groupCount,
            IsMoe = isMoe,
            NumExperts = numExperts,
            NumExpertsPerTok = usedCount,
            MoeFfnSize = expFfn,
            MoeSharedFfnSize = sharedFfn,
            MoeSharedCount = sharedCount,
            ExpertWeightsScale = expScale,
            VocabSize = (int)gguf.GetUint32($"{arch}.vocab_size", 0),
            ContextLength = (int)gguf.GetUint32($"{arch}.context_length", 4096),
            Eps = gguf.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f),
            IsRecurrent = isRecurrent,
        };
    }
}
