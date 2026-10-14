using Flygon.Gguf;

namespace Flygon.Model;

/// <summary>
/// Qwen3.5 is a hybrid transformer: some layers use gated-DeltaNet linear
/// attention, the rest use standard full attention. This config captures the
/// two families of dimension parameters plus which layers are linear.
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

        // Linear layers are the odd ones: every `full_attention_interval`th layer
        // (1-indexed) is full attention, the rest are linear.
        int layers = (int)gguf.GetUint32($"{arch}.block_count");
        int fullAttnInterval = gguf.GetInt32($"{arch}.full_attention_interval", 4);
        bool[] isRecurrent = new bool[layers];
        for (int i = 0; i < layers; i++)
            isRecurrent[i] = (i + 1) % fullAttnInterval != 0;

        return new ModelConfig
        {
            Architecture = arch,
            NumLayers = layers,
            HiddenSize = hidden,
            FfnSize = (int)gguf.GetUint32($"{arch}.feed_forward_length"),
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
            VocabSize = (int)gguf.GetUint32($"{arch}.vocab_size", 0),
            ContextLength = (int)gguf.GetUint32($"{arch}.context_length", 4096),
            Eps = gguf.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f),
            IsRecurrent = isRecurrent,
        };
    }
}
