# ASAP ICASSP paper draft

This folder contains the working draft of the ASAP paper, targeting **ICASSP 2026**.
Each section is kept in its own Markdown file so that drafting, review, and Git history stay focused per section. The final LaTeX manuscript will be assembled later from these Markdown drafts.

## Working title

**ASAP: Adaptive Spherical Anomaly-Guided Purification Against LiDAR Point Cloud Attacks**

## Target venue and format

- **Venue**: IEEE ICASSP 2026 (regular paper).
- **Format**: IEEE two-column conference template.
- **Page budget**: typically 4 content pages + 1 page for references (subject to the final ICASSP 2026 call).

## File layout

| File | Section | Target rough length on the final paper |
|------|---------|----------------------------------------|
| `00_abstract.md`     | Abstract                             | 150 - 200 words           |
| `01_introduction.md` | 1. Introduction                      | about 3/4 of column 1     |
| `02_related_work.md` | 2. Related Work                      | about 1/2 column          |
| `03_method.md`       | 3. ASAP Framework                    | 1.5 - 2 columns + figure  |
| `04_experiments.md`  | 4. Experiments                       | 1 - 1.5 columns           |
| `05_conclusion.md`   | 5. Conclusion                        | about 1/4 column          |
| `detector_selection.md` | Detector selection notes          | internal planning note    |
| `experiments_overview.html` | Unified experiment-ID system + status board | internal planning note (open in browser) |
| `references.bib`     | References (BibTeX)                  | grows as we cite          |

## Section status

| Section            | Status        | Notes                                      |
|--------------------|---------------|--------------------------------------------|
| Abstract           | first draft   | written before experimental numbers exist  |
| Introduction       | first draft   | 1.1-1.4 in prose; cites attacks/datasets/diffusion baselines |
| Related work       | first draft   | 2.1 attacks / 2.2 defenses / 2.3 diffusion purification     |
| Method             | first draft   | full math derivations for M1/M2/M3 + Algorithm 1 + complexity |
| Experiments        | first draft   | 4.1-4.5 in prose; Tables 1-3 with [XX.X] placeholders pending experiments |
| Detector selection | planning note | DSVT-guided modern detector plan           |
| Experiments overview | planning note | E0-E6 experiment IDs, MS1-MS4 milestones, status board |
| Conclusion         | first draft   | written; numerical claims [XX.X] pending   |
| References         | seeded        | dataset / attack / defense / diffusion stubs added; all marked UNVERIFIED |

## Writing conventions

- Write all paper content in **English**, since the final manuscript will be English.
- Use Chinese only inside `<!-- ... -->` HTML comments for internal notes, and remove all comments before submission.
- Method names are bold and capitalized in their first appearance, e.g. **Adaptive Radius**, **Anomaly Scorer**, **Selective Diffusion**, and abbreviated as M1, M2, M3 thereafter.
- Equations and symbols stay consistent across sections; symbols are defined once in `03_method.md` and reused.
- Numerical results are written as `TBD`, `TODO`, or `[xx.x]` until the corresponding experiment is logged.
- All citations use BibTeX keys in `references.bib`; in Markdown drafts, cite as `[@bibkey]`.
- Keep each section under its rough length budget so that the final paper still fits 4 columns.

## How to update

- Edit only one section per commit when possible, so reviewers can read the diff cleanly.
- When numerical results land, replace the placeholder in `00_abstract.md` and the corresponding table/paragraph in `04_experiments.md` in the same commit.
- Keep the status table in this README in sync when a section moves from `skeleton` to `first draft`, `revised`, or `final`.
