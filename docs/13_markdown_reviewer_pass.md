# ASAP Markdown Reviewer Pass — 2026-06-02

## Scope

This pass reviews the consolidated Markdown manuscript under `docs/paper/` before LaTeX assembly. It focuses on ICASSP reviewer risk: novelty, experiment sufficiency, threat-model clarity, table readability, and claim control.

## Simulated Decision

**Borderline / weak accept if claims remain conservative; weak reject if the paper implies broad SOTA dominance.**

The current paper is defensible as an inference-only, detector-agnostic LiDAR purification framework. Its strongest evidence is not universal superiority over ROR, but the combination of:

- detector transfer across PointPillars, PV-RCNN, VoxelNeXt, and TransFusion-Lidar;
- M2 selection preventing all-unit diffusion collapse;
- ASAP-pfilter slightly improving over the same support rule used by ROR on both nuScenes perturbation detectors;
- explicit caveats for Dropping and keyframe-only nuScenes attacks.

## Findings And Actions

| Area | Reviewer concern | Current status | Action |
|------|------------------|----------------|--------|
| Novelty vs ROR | pfilter may look like ROR with extra steps. | The caveat is present in Method and Experiments. | Keep wording as "score-net + support-filter cascade"; avoid saying diffusion alone beats ROR. |
| M1 adaptive radius | Fixed `r=0.40` slightly exceeds adaptive Track A. | Introduction and Experiments now frame M1 as conservative locality, not AP dominance. | Keep the negative diagnostic visible but short. |
| M2 feature necessity | `only_f_d` nearly matches full M2 on KITTI Perturbation. | Paper now frames M2 as cross-attack coverage. | Do not claim all four features are indispensable for perturbation. |
| External baselines | LiDAR-SPD, PointDP, Ada3Diff, ScAR, PCLD are not fully reproduced. | Related work cites representative diffusion and local-SPU methods. | Add boundary that these methods differ in task, training regime, or full-scene availability; use SPD-style proxy as internal stress test only. |
| Threat model | nuScenes attacks are keyframe-only under ten-sweep evaluation. | Experiments disclose this and exclude Dropping. | Keep caveat in table caption/text and limitation sentence. |
| Runtime | ASAP is slower than ROR. | Experiments state prototype status and 0.48 s/frame. | Do not claim real-time deployment. |
| Table width | Table 1 and Table 2 are too wide for one column. | Markdown is clear, LaTeX will need layout decisions. | Use two-column table or compressed notation during LaTeX stage. |

## Edits Applied In This Pass

1. Abstract: changed the final result sentence to say ASAP-pfilter "slightly exceeds" ROR and explicitly calls it a cascade.
2. Related Work: added an external-method boundary sentence covering LiDAR-SPD, PointDP, Ada3Diff, ScAR, PCLD, and similar methods.
3. Conclusion: replaced "stronger-than-ROR" with a more precise pfilter-cascade statement.

## Remaining Before LaTeX

1. Decide Table 1 layout: likely two-column table with shortened detector-family labels.
2. Decide Table 2 layout: keep only the current 12 rows or split KITTI / nuScenes ablations.
3. Convert Markdown equations and Algorithm 1 carefully, preserving equation numbers and pfilter caveat.
4. Do not update `docs/paper/main_icassp.tex` until the user explicitly moves to LaTeX.
