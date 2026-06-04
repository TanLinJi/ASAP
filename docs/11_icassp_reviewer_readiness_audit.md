# ICASSP Reviewer Readiness Audit — 2026-06-01

## Scope

This audit reviews the current ASAP draft as an ICASSP-style submission package. It uses the completed KITTI and nuScenes experiments, the P0/P1/P2/P4 ablations, the completed E5.1/E5.2 corrective ablations, and the current Markdown paper sections under `docs/paper/`.

## Simulated Editorial Decision

**Borderline / weak accept if the final paper is tightly written; borderline / weak reject if claims are overstated.**

The current evidence is now strong enough for a defensible submission, but not for a broad SOTA-dominance paper. The safest submission story is:

> ASAP is an inference-only, detector-agnostic LiDAR purification framework whose central contribution is local selective purification with an anomaly-selective structure-preservation gate. It is competitive with strong support filtering, improves over nonselective diffusion-style proxies, and transfers across modern nuScenes detectors under accepted keyframe-only attacks.

## Reviewer 1 — Method Novelty

**Likely concern.** The pfilter rows may look too close to ROR because both use a radius-support rule.

**Current defense.** The paper now explicitly labels pfilter as a score-net + support-filter cascade. The support-only/ROR baseline is reported with the same `0.4 m / 5 neighbors` rule, and pfilter adds about +0.9 mAP over that rule on both VoxelNeXt and TransFusion-Lidar.

**Required revision.** Keep the pfilter caveat visible in the abstract, main table caption, method Section 3.5.5, and experiments Section 4.3. Do not describe nuScenes Perturbation as “diffusion alone beats ROR.”

## Reviewer 2 — Ablation Sufficiency

**Likely concern.** Is M2 just a runtime optimization, or does it change robustness?

**Current defense.** P2 is strong: adaptive-SPU all-unit diffusion collapses to **38.62 / 17.88**, while ASAP Track A reaches **61.04 / 34.69**. P0 also shows fixed-SPU all-unit diffusion is weaker and much heavier than ASAP. The E5.2 feature ablation adds an important boundary: density ratio `f_d` is the dominant KITTI Perturbation cue, and `only_f_d` nearly matches Track A at detector level. Therefore the paper should not claim that all four M2 features are indispensable for perturbation.

**Required revision.** In the final manuscript, Table 2 should emphasize “structure preservation” rather than “compute allocation.” M2 should be described as a compact cross-attack analytic gate, with `f_d` carrying most of the perturbation signal in the current KITTI setting.

## Reviewer 2b — Adaptive Radius Claim

**Likely concern.** Does M1 actually improve AP, or is a fixed radius enough?

**Current defense.** E5.1 shows fixed `r=0.40` reaches **61.06 / 34.92**, slightly above adaptive Track A's **61.04 / 34.69**, but edits more points (**27.07%** vs **21.93%**). Smaller fixed radii remain near the no-defense lower bound.

**Required revision.** Do not claim adaptive radius strictly dominates every fixed radius. Present M1 as a conservative locality prior that avoids hand-picking a large global radius and reduces unnecessary edits while retaining near-identical robustness.

## Reviewer 3 — Experimental Validity

**Likely concern.** nuScenes attacks are keyframe-only while OpenPCDet loads ten sweeps.

**Current defense.** The table caption and experiments text disclose this. Dropping is excluded because the attack-strength gate failed even at `drop_ratio=1.0`.

**Required revision.** Keep the keyframe-only caveat in the main table caption and limitation paragraph. Do not use wording that implies full temporal-sweep robustness.

## Devil's Advocate

**Strongest rejection argument.** The direct baselines are mostly SOR/ROR; recent methods such as LiDAR-SPD, ScAR, PointDP, PCLD, and Ada3Diff are not fully reproduced under the same full-scene detection setting.

**Best response.** Position external methods carefully: many are training-time, classification/object-level, corruption-focused, or lack directly usable full-scene LiDAR detection code. The SPD-style fixed spherical diffusion proxy should be framed as an internal design stress test, not as an official LiDAR-SPD reproduction.

**Action.** Related work should state this boundary explicitly. A short “baseline feasibility” note can be included in the appendix/internal response material, but the main paper should avoid claiming exhaustive SOTA comparison.

## Highest-Priority Edits Before LaTeX Assembly

1. Tighten the abstract: keep one quantitative nuScenes Perturbation result, one sentence on detector-agnostic transfer, and one caveat that pfilter is a disclosed cascade.
2. Compress Section 3: ICASSP page budget is tight. Preserve formulas for M1, M2 aggregation, M3 local purification, and pfilter; move long explanatory paragraphs into shorter prose.
3. Ensure Table 2 fits one column or decide to make it a two-column table. Its current content is important, but it may be too wide in final IEEE format.
4. Add a short limitation sentence in Conclusion: “ASAP does not reconstruct evidence removed by strong dropping attacks.”
5. Keep runtime claims conservative: prototype cost is higher than ROR, and the current claim is robustness/transfer, not real-time deployment.

## Go / No-Go

**Go for Markdown manuscript consolidation, not LaTeX yet.** The next mainline step should be tightening the current paper Markdown into an ICASSP-length narrative with controlled claims. New GPU experiments should wait unless this consolidation reveals a concrete acceptance-critical gap.
