using System.Text;

namespace Flygon.Model;

public readonly record struct ChatMessage(string Role, string Content);

public static class ChatTemplate
{
    /// <summary>
    /// Qwen3.5 Jinja (text-only). The assistant prompt opens a <think> block
    /// by default; pass <paramref name="enableThinking"/>(false) to emit
    /// an empty think block instead.
    /// </summary>
    public static string RenderQwen35(
        IReadOnlyList<ChatMessage> messages,
        bool addGenerationPrompt = true,
        bool enableThinking = true)
    {
        StringBuilder sb = new();
        int startIdx = 0;
        if (messages.Count > 0 && messages[0].Role == "system")
        {
            sb.Append("<|system|>\n");
            sb.Append(messages[0].Content.Trim());
            sb.Append("\n<|/system|>\n");
            startIdx = 1;
        }

        for (int i = startIdx; i < messages.Count; i++)
        {
            ChatMessage msg = messages[i];
            if (msg.Role == "user")
            {
                sb.Append("<|user|>\n");
                sb.Append(msg.Content.Trim());
                sb.Append("\n<|/user|>\n");
            }
            else if (msg.Role == "assistant")
            {
                sb.Append("<|assistant|>\n");
                sb.Append(msg.Content);
                sb.Append("<|/assistant|>\n");
            }
        }

        if (addGenerationPrompt)
        {
            sb.Append("<|assistant|>\n");
            if (enableThinking)
                sb.Append("<think>\n");
            else
                sb.Append("<think>\n\n</think>\n\n");
        }

        return sb.ToString();
    }
}
