# Dogfood Golden Sample: Baidu Homepage

This dataset incorporates the real dogfood run against <https://www.baidu.com/> on 2026-04-15 into the self-evolution sample set for the `dogfood` skill.

## Source Artifacts

- Source report: `datasets/skills/dogfood/baidu-homepage/source_report.md`
- Dataset directory: `datasets/skills/dogfood/baidu-homepage`

## What This Sample Covers

- Homepage load health and console cleanliness
- Search submission flow
- Search suggestion relevance
- Wenxin assistant entry and back-navigation chain
- Top-nav News entry health

## Why It Matters

This sample gives `dogfood` a real browser-heavy golden set with both:

- **positive paths**: homepage load, Wenxin single-turn QA, News page load
- **negative/blocking paths**: search flow interrupted by Baidu security verification, unrelated suggestions, unstable back-navigation chain

That makes it more useful than a purely synthetic sample when evaluating whether the evolved skill:

1. tests the intended user flows,
2. distinguishes blockers from non-blockers,
3. captures evidence correctly, and
4. writes a balanced QA report with both working and broken paths.

## Notes

- The source report is stored in-repo as text only; screenshot binaries from the original run are intentionally not committed.
