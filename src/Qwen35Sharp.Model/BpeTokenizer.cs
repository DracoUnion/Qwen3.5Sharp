using System.Text;
using System.Text.RegularExpressions;
using Flygon.Gguf;

namespace Flygon.Model;

/// <summary>
/// Byte-level BPE tokenizer driven by <c>tokenizer.ggml.*</c> metadata.
/// Qwen3.5 special-token detection stops on <c>eos_token_id</c> and the
/// thinking end tag <c></s></c>.
/// </summary>
public sealed class BpeTokenizer
{
    private readonly string[] _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<string, int> _merges;
    private readonly HashSet<int> _specialIds;
    private readonly Regex _pretokenizer;

    public int BosId { get; }
    public int EosId { get; }
    public int UnkId { get; }
    public int VocabSize => _idToToken.Length;

    public BpeTokenizer(GgufFile gguf)
    {
        string[] tokens = gguf.GetStringArray("tokenizer.ggml.tokens");
        if (tokens.Length == 0)
            throw new InvalidDataException("GGUF missing tokenizer.ggml.tokens");

        _idToToken = tokens;
        _tokenToId = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
            _tokenToId.TryAdd(tokens[i], i);

        _merges = new Dictionary<string, int>(StringComparer.Ordinal);
        string[] merges = gguf.GetStringArray("tokenizer.ggml.merges");
        for (int i = 0; i < merges.Length; i++)
            _merges.TryAdd(merges[i], i);

        BosId = gguf.GetInt32("tokenizer.ggml.bos_token_id", 0);
        EosId = gguf.GetInt32("tokenizer.ggml.eos_token_id", 0);
        UnkId = gguf.GetInt32("tokenizer.ggml.unknown_token_id", 0);

        int[] tokenTypes = gguf.GetInt32Array("tokenizer.ggml.token_type");
        _specialIds = new HashSet<int>();
        for (int i = 0; i < tokens.Length; i++)
        {
            bool typed = i < tokenTypes.Length && (tokenTypes[i] == 3 || tokenTypes[i] == 4);
            if (typed || i == BosId || i == EosId)
                _specialIds.Add(i);
        }

        string pre = gguf.GetString("tokenizer.ggml.pre", "gpt2");
        _pretokenizer = new Regex(ResolvePreTokenizer(pre), RegexOptions.Compiled);
    }

    public int[] Encode(string text)
    {
        List<(string Text, int[]? Ids)> fragments = [(text, null)];
        // Pre-split known special tokens so they survive BPE merging.
        foreach (int id in _specialIds)
        {
            string special = _idToToken[id];
            if (special.Length == 0)
                continue;
            List<(string Text, int[]? Ids)> next = [];
            foreach ((string frag, int[]? presetIds) in fragments)
            {
                if (presetIds != null)
                {
                    next.Add((frag, presetIds));
                    continue;
                }

                int start = 0;
                while (true)
                {
                    int idx = frag.IndexOf(special, start, StringComparison.Ordinal);
                    if (idx < 0)
                    {
                        if (start < frag.Length)
                            next.Add((frag[start..], null));
                        break;
                    }
                    if (idx > start)
                        next.Add((frag[start..idx], null));
                    next.Add((special, [id]));
                    start = idx + special.Length;
                }
                fragments = next;
            }
        }

        List<int> ids = [];
        foreach ((string frag, int[]? preset) in fragments)
        {
            if (preset != null)
            {
                ids.AddRange(preset);
                continue;
            }

            foreach (Match match in _pretokenizer.Matches(frag))
            {
                string normalized = ByteEncode(match.Value);
                if (_tokenToId.TryGetValue(normalized, out int direct))
                {
                    ids.Add(direct);
                    continue;
                }
                ids.AddRange(Bpe(normalized));
            }
        }

        return [.. ids];
    }

    public bool IsStop(int id) => id == EosId || id < 0;

    public bool TryLookup(string token, out int id) => _tokenToId.TryGetValue(token, out id);

    public string Decode(IReadOnlyList<int> ids)
    {
        List<byte> bytes = new(ids.Count);
        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if ((uint)id >= (uint)_idToToken.Length)
                continue;
            AppendTokenBytes(_idToToken[id], bytes);
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    public string DecodeVisible(IReadOnlyList<int> ids) => Decode(ids);

    private List<int> Bpe(string token)
    {
        if (token.Length == 0)
            return [];
        List<string> word = new(token.Length);
        for (int i = 0; i < token.Length; i++)
            word.Add(token[i].ToString());

        while (word.Count > 1)
        {
            int best = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i < word.Count - 1; i++)
            {
                if (_merges.TryGetValue(word[i] + " " + word[i + 1], out int rank) && rank < best)
                {
                    best = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
                break;
            word[bestIndex] = word[bestIndex] + word[bestIndex + 1];
            word.RemoveAt(bestIndex + 1);
        }

        List<int> ids = new(word.Count);
        for (int i = 0; i < word.Count; i++)
        {
            if (_tokenToId.TryGetValue(word[i], out int id))
                ids.Add(id);
            else
                ids.Add(UnkId);
        }

        return ids;
    }

    internal static string ResolvePreTokenizer(string preTokenizerType) => preTokenizerType switch
    {
        // Qwen / GPT-2 style: standard byte-level BPE pretokenizer.
        _ =>
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    };

    private static string ByteEncode(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        char[] chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
            chars[i] = EncodeByte(bytes[i]);
        return new string(chars);
    }

    private static char EncodeByte(byte b)
    {
        if (b == 0xAD)
            return (char)0x0143;
        if (b <= 0x20)
            return (char)(b + 0x0100);
        if (b >= 0x7F && b <= 0xA0)
            return (char)(b + 0x00A2);
        return (char)b;
    }

    private static void AppendTokenBytes(string token, List<byte> bytes)
    {
        foreach (char r in token)
        {
            if (r == 0x0100)
                continue;
            if (r > 0x0100 && r <= 0x0120)
            {
                bytes.Add((byte)(r - 0x0100));
                continue;
            }

            if (r >= 0x0121 && r <= 0x0142)
            {
                bytes.Add((byte)(r - 0x00A2));
                continue;
            }

            if (r == 0x0143)
            {
                bytes.Add(0xAD);
                continue;
            }

            if (r == 0x2581)
            {
                bytes.Add(0x20);
                continue;
            }

            if (r < 0x100)
            {
                bytes.Add((byte)r);
                continue;
            }

            bytes.AddRange(Encoding.UTF8.GetBytes([r]));
        }
    }
}
