# ALFWorld E1 rerun (2026-09-05)

This is a three-session `valid_seen` smoke experiment, not the final E1 sample.
Each session uses a newly generated GLM-5.3-Flash task-level hint per dose and
reuses that hint at all decision turns. Targets are the ALFWorld expert actions,
not L3 Teacher actions. The frozen Student is Qwen3-4B.

Scoring uses exactly three teacher-forced views: public state, public state plus
hint, and hint only. Token means are averaged across turns and then across the
three sessions with equal session weight. The raw 321-row trace was retained as
a server artifact but is not committed because it is 655 KB.

The behavioral separation works: L2 queries before pickup in every session and
never directly hits the hidden source, while L3 directly hits it in every
session and fact-leaks in every generated hint. The analytical copy sanity check
fails: mean copy is L1 0.9643, L2 0.6996, and L3 0.2627. Therefore the current
positive-per-token copy statistic is dominated by generic command/template
tokens and must not be used as a plagiarism reward or paper metric yet.
