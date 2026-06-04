# ASAP SOTA Baseline Expansion Plan — 2026-06-02

## Goal

Add credible recent SOTA comparisons beyond SOR/ROR. The target is not to overclaim broad SOTA dominance, but to show that ASAP improves over or is competitive with recent defense ideas under a transparent, same-protocol comparison.

## Candidate Set

| Candidate | Year | Task fit | Code status | Same-protocol experiment? | Decision |
|-----------|-----:|----------|-------------|---------------------------|----------|
| LiDAR-SPD | 2025 | Direct LiDAR 3D detection defense with spherical purification units | PDF available locally; workspace has only a simplified single-file concept script, not official runnable code | Yes, via fixed spherical all-unit diffusion proxy | **Completed proxy** |
| Towards Real-Time Defense against Object-Based LiDAR Attacks | 2025 | Direct LiDAR object-detection defense, model-agnostic, real-time | Paper found; no public code found in initial search | Not yet, unless code is released or algorithm is reimplemented | **Cite / feasibility note** |
| PWAVEP | 2026 | Input-level point-cloud adversarial purification via spectral graph wavelets | Official GitHub cloned under `third_party/external_baselines/pwavep` | No direct KITTI detection baseline: current code is ModelNet/ShapeNet classification-oriented and uses classifier gradients / labels | **Cite / not same-protocol** |
| APC | 2026 | Point-cloud recognition defense / counterattack candidate | Claimed code link not reachable in initial search | No | **Defer** |
| KNN-Defense | 2025 | Point-cloud classification feature-space KNN voting | Code cloned | No: does not output purified point clouds and needs ModelNet classifier features | **Not a KITTI baseline** |

## Completed Experiment: LiDAR-SPD-Style Proxy

LiDAR-SPD reports `r1=0.15m, r2=0.10m`. We ran a fixed spherical all-unit diffusion proxy with:

- KITTI PointPillars E2.2 Perturbation.
- `FIXED_RADIUS=0.15`, `INNER_BETA=0.6667`, `STRIDE=0.15`.
- M2 disabled with `tau=-1.0`.
- Same Track A score-net and `score_anchor_blend=0.75`.
- tmux session: `asap_spd_r015_e22_20260602`.
- Output root: `outputs/spd_style_uniform_score_net_r015/`.

PointPillars result:

- **56.58 / 26.61** Car Moderate AP_R40 / Moderate mAP on KITTI PointPillars E2.2 Perturbation.
- 100.00% SPUs flagged, 95.57% SPUs score-net processed, and 64.40% points edited.
- Comparison anchors under the same protocol: no defense **56.54 / 27.32**; ASAP current E2.2 **61.26 / 34.76**.

PV-RCNN transfer result on the same purified scans:

- **3.16 / 2.31** Car Moderate AP_R40 / Moderate mAP on KITTI PV-RCNN E2.2 Perturbation.
- Comparison anchors under the same protocol: no defense **3.50 / 2.45**; ROR **1.94 / 1.17**; ASAP current E2.2 **11.23 / 6.26**.

Boundary:

This is **not** an official LiDAR-SPD reproduction. LiDAR-SPD also includes spherical projection and a conditioned diffusion model. The proxy isolates the closest same-protocol design idea: fixed spherical all-unit diffusion under the ASAP/OpenPCDet attack and detector setup.

Decision:

Use this as the main recent-method proxy comparison. It supports ASAP's central selectivity claim because the fixed all-unit design remains close to or below no defense and below the current ASAP E2.2 candidate while editing almost three times as many points as Track A. The PV-RCNN transfer result rules out the concern that the proxy only underperforms on PointPillars.

Additional local check:

`/root/autodl-tmp/LiDAR_SPD/lidar-spd.py` is a simplified concept script: it randomly builds spherical units, instantiates an untrained placeholder diffusion model, and returns points unchanged in `spherical_diffusion`. It is not suitable as official LiDAR-SPD reproduction code.

## PWAVEP 2026 Feasibility

The cloned PWAVEP code exposes a purifier, but its runnable path is tied to classification:

- `protocols/defenders/run_pwavep.py` loads a ModelNet40/ShapeNetPart classifier and attacked `.pt` data.
- `src/defenders/wavelet_def/starter.py` calls `extract_grad_and_risk(...)`, requiring labels and gradients from the victim classification model.
- `WaveletTransformUtil.extract_grad_wrt_coffs(...)` computes cross-entropy gradients through the target model.

This is not a standalone full-scene KITTI purifier and does not output `.bin` scans for OpenPCDet. Adapting it would require defining detection-specific risk scores or replacing the official classifier-gradient saliency path with a model-free proxy, which would no longer be an official PWAVEP reproduction. Therefore PWAVEP should be cited as a 2026 point-cloud recognition purifier, not used as a same-protocol KITTI detection baseline in the main table.

## 2025/2026 Baseline Strategy

Minimum acceptable comparison set:

1. **LiDAR-SPD-style proxy (2025)**: same detector, same attack, same metric. Completed and favorable to ASAP.
2. **CCS 2025 real-time LiDAR defense**: cite as recent direct LiDAR detection defense; explain that no same-protocol reproduction is claimed because public runnable code was not found.
3. **PWAVEP 2026**: cite as recent point-cloud purification SOTA, but keep it out of the main detection table because the official code path is classification-oriented.

## Implementation Notes

- Do not put external reported numbers into the main result table unless the attack, detector, dataset, and metric match.
- If using a proxy, name it explicitly as a proxy in both table and caption.
- Use tmux for all full KITTI/nuScenes runs.
- Use both T4 GPUs where the baseline can shard by frame.

## Immediate Next Actions

1. Update `docs/paper/04_experiments.md` with the completed LiDAR-SPD-style proxy row.
2. Add BibTeX entries for CCS 2025 and PWAVEP 2026 if they appear in the main text.
3. Next experiment should target a genuinely same-protocol external comparison only if it can output KITTI/nuScenes full-scene purified point clouds. Otherwise, prioritize strengthening ASAP's own detector-transfer and attack-transfer evidence, because the current recent-method proxy comparison is already favorable.
