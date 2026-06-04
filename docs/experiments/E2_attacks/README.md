# E2 — Adversarial attacks

Each E2 experiment reproduces one attack family and produces an "attacked val set" stored under `outputs/attacks/...`. Downstream defenses (E3 / E4) consume those attacked sets.

| ID | Attack | Category | File | Status |
|----|--------|----------|------|--------|
| E2.1 | Point Injection     | injection    | [`E2.1_point_injection.md`](E2.1_point_injection.md)       | **done for KITTI + nuScenes keyframe-only attack** |
| E2.2 | Point Perturbation  | perturbation | [`E2.2_point_perturbation.md`](E2.2_point_perturbation.md) | **done for KITTI + nuScenes keyframe-only attack** |
| E2.3 | Point Dropping      | dropping     | [`E2.3_point_dropping.md`](E2.3_point_dropping.md)         | **done for KITTI + nuScenes keyframe-only attack** |

Conventions:
- Each E2 file documents the attack budget (point count, $\ell_p$ radius, drop ratio) explicitly so that E3/E4 rows are reproducible.
- Attacked point clouds are stored alongside an `attack_meta.jsonl` log so that no run-time randomness is needed at evaluation time.
- The same attack is run **per detector** when targeted attacks need the victim's gradients; this is recorded inside each E2 file's *Setup* section.
- Current KITTI E2 outputs are detector-agnostic geometric/random attacks under `outputs/attacks/kitti/E2.{1,2,3}_*/velodyne_attacked/`, each with 3769 KITTI val bins. Do not describe them as gradient-optimized PGD or saliency attacks unless a stronger detector-aware variant is later implemented.
- Current nuScenes E2 outputs are detector-agnostic **keyframe-only** attacks under `outputs/attacks/nuscenes_keyframe/E2.{1,2,3}_*/samples/LIDAR_TOP/`, each with 6019 val keyframe `.pcd.bin` files. OpenPCDet still loads clean historical sweeps because the default nuScenes configs use `MAX_SWEEPS=10`; paper text must call this the keyframe-only nuScenes protocol. The accepted nuScenes attack set for main rows is Injection with the original budget plus Perturbation with the budget-gated `epsilon=0.5` setting under `outputs/attacks/nuscenes_keyframe_budget_gate/E2.2_perturbation_eps0p5/`. Dropping remains paused for nuScenes main claims because even keyframe-only `drop_ratio=1.0` reaches only 8.2% mAP ASR on VoxelNeXt.
