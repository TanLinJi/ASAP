# ASAP documentation index

This `docs/` directory hosts every human-readable document for the ASAP project (Adaptive Spherical Anomaly-Guided Purification, ICASSP 2026 submission).

## Top-level layout

| Path | Purpose | Status |
|------|---------|--------|
| `01_SA-SPD_proposal.html`     | Long-form Chinese research proposal (background, modules, timeline, risk). | Stable historical proposal |
| `02_implementation_plan.md`   | Engineering implementation plan: data layout, OpenPCDet backend, pipeline stages. | Living document |
| `03_icassp_paper_workflow.md` | End-to-end ICASSP paper workflow from claim freeze to experiments, writing, integrity check, review, and final formatting. | Active control plan |
| `09_icassp_acceptance_gap_audit.md` | Submission-risk audit focused on claim support, reviewer objections, and next actions for acceptance. | Active |
| `10_innovation_and_sota_upgrade_plan.md` | Method novelty and SOTA-baseline upgrade plan, including the SPD-style uniform diffusion baseline. | Active |
| `11_icassp_reviewer_readiness_audit.md` | Simulated ICASSP reviewer-readiness audit with go/no-go decision and highest-priority edits. | Active |
| `15_post_tstar_reviewer_experiment_decision.md` | Post-`t_star` reviewer decision: closes the KITTI micro-sweep, freezes main experiment rows, and prioritizes reviewer-readiness work. | Active |
| `16_score_net_loss_upgrade_plan.md` | Planned score-net loss-function upgrade experiments for stronger M3 evidence. | Active |
| `paper/`                      | ICASSP 2026 paper draft, one Markdown file per section, plus verified BibTeX and LaTeX-ready figure/table fragments. | Revised draft, ready for first LaTeX assembly |
| `experiments/`                | **Per-experiment records.** One sub-folder per experiment family (E0..E6), one Markdown file per experiment (E0.1, E1.2, ...). | Skeletons in place, results to be filled |
| `references/`                 | Reference PDFs and reading list. | Manually curated |

## How the documents reference each other

- **Method modules** are named `M1` (Adaptive Radius), `M2` (Anomaly Scorer), `M3` (Selective Diffusion). They are defined in `paper/03_method.md` and consumed everywhere else.
- **Experiments** are named `E0..E6` with sub-IDs like `E0.3`, `E4.1`. The full list lives in `experiments/README.md`; each ID has its own dedicated record in `experiments/<family>/E<id>_<name>.md`.
- **Milestones** are named `MS1..MS4` and grouped in `experiments/README.md` together with the experiment IDs they include.
- **Paper execution workflow** lives in `03_icassp_paper_workflow.md`; use it as the control plan before changing experiment scope or paper claims.

## Writing conventions

- Paper-facing content (in `paper/`) is **English only**; internal planning notes can use Chinese inside `<!-- ... -->` HTML comments.
- Engineering/log content (in `experiments/`, `02_implementation_plan.md`, this index) may be written in **English with Chinese commentary**, since the audience is the project team.
- Always cite a specific file/line using the absolute path, e.g. `/root/autodl-tmp/ASAP/docs/paper/03_method.md`.
- When introducing a new experiment, **first** add an entry to `experiments/README.md` (status board), **then** create the per-experiment Markdown file from `experiments/_template.md`.

## Adding a new document

1. Pick the right family: paper section (`paper/`), per-experiment record (`experiments/<family>/`), or engineering plan (`02_implementation_plan.md`).
2. For experiments, copy `experiments/_template.md` to `experiments/<family>/E<id>_<short_name>.md`, fill the required fields, and link it from the family-level `README.md`.
3. Keep one canonical home for every piece of information; cross-link with absolute paths instead of duplicating prose.
