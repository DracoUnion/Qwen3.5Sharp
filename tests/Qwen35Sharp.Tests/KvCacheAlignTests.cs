using Flygon.Model;

namespace Flygon.Tests;

public sealed class KvCacheAlignTests
{
    [Fact]
    public void DivergingTail_KeepsSharedPrefix()
    {
        int[] cached = [1, 2, 3, 4, 5];
        int[] prompt = [1, 2, 3, 6, 7];
        PromptReuse plan = KvCacheAlign.Plan(cached, prompt);
        Assert.Equal(3, plan.TruncateTo);
        Assert.Equal(3, plan.SuffixStart);
        Assert.Equal([6, 7], prompt[plan.SuffixStart..]);
    }

    [Fact]
    public void LongerCacheThanPrompt_DoesNotDropPrefix()
    {
        int[] cached = [1, 2, 3, 10, 11];
        int[] prompt = [1, 2, 3, 6];
        PromptReuse plan = KvCacheAlign.Plan(cached, prompt);
        Assert.Equal(3, plan.TruncateTo);
        Assert.Equal([6], prompt[plan.SuffixStart..]);
    }

    [Fact]
    public void FullHit_ReplaysLastToken()
    {
        int[] ids = [1, 2, 3];
        PromptReuse plan = KvCacheAlign.Plan(ids, ids);
        Assert.Equal(2, plan.TruncateTo);
        Assert.Equal(2, plan.SuffixStart);
        Assert.Equal([3], ids[plan.SuffixStart..]);
    }

    [Fact]
    public void EmptyCache_ForwardsWholePrompt()
    {
        int[] prompt = [4, 5, 6];
        PromptReuse plan = KvCacheAlign.Plan([], prompt);
        Assert.Equal(0, plan.TruncateTo);
        Assert.Equal(prompt, prompt[plan.SuffixStart..]);
    }

    [Fact]
    public void NoOverlap_Resets()
    {
        PromptReuse plan = KvCacheAlign.Plan([1, 2, 3], [9, 8]);
        Assert.Equal(0, plan.TruncateTo);
        Assert.Equal(0, plan.SuffixStart);
    }
}
