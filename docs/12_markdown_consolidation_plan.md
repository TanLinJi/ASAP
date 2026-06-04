# ASAP Markdown Consolidation Plan — 2026-06-02

## Goal

Compress the current Markdown paper into an ICASSP-length narrative before any LaTeX work. The experiment records remain the source of full detail; the paper draft should only carry claims and numbers needed for a defensible 4-page submission.

## Current Length Snapshot

| Section | Current words | Target words | Action |
|---------|--------------:|-------------:|--------|
| Abstract | 176 | 150-180 | Keep. |
| Introduction | 661 | 550-650 | Keep, with conservative M1/M2 claims. |
| Related Work | 445 | 350-450 | Keep. |
| Method | 2787 | 1200-1600 | Compress formulas and remove long explanatory derivations. |
| Experiments | 2325 | 1100-1400 | Compress setup, table commentary, ablation narrative, and cost. |
| Conclusion | 133 | 100-140 | Keep. |

## Main-Text Evidence Policy

Keep in the paper:

- Main detector table across KITTI and nuScenes.
- pfilter caveat: score-net + support-filter cascade, not pure diffusion.
- M2 selection vs all-unit diffusion, because this is the strongest innovation ablation.
- A short negative diagnostic for M1 and M2 feature subsets to prevent overclaiming.
- Runtime caveat: prototype is slower than ROR; claim is robustness/transfer, not real-time.

Move to experiment records only:

- Full M1 radius sweep details.
- Full M2 feature AUC table.
- Tau sweep details beyond the final conclusion.
- Qualitative per-frame selectivity statistics unless used as a figure caption.

## Next Implementation Order

1. Compress `docs/paper/04_experiments.md`. **Done**: 2325 -> 1131 words.
2. Compress `docs/paper/03_method.md`. **Done**: 2787 -> 850 words.
3. Rerun `scripts/check_paper_integrity.py`. **Done**: passed.
4. Run a reviewer-style Markdown-only pass before LaTeX assembly. **Done**: see `docs/13_markdown_reviewer_pass.md`.
