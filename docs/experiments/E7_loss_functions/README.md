# E7 — Score-net Loss Function Upgrades

Loss-function experiments for the ASAP local VP-SDE score-net. These experiments are not part of the current frozen main table until a variant passes both PointPillars screening and PV-RCNN confirmation.

| ID | Target | Description | File | Status |
|----|--------|-------------|------|--------|
| E7.1 | VP-SDE score-net loss | Time-balanced DSM, geometry consistency, density preservation, and paired attacked-clean fine-tuning | [`E7.1_score_net_loss_upgrade.md`](E7.1_score_net_loss_upgrade.md) | **planned** |

Decision rule:

- Keep L0 (`asap_score_net_trackA_v1.pth`) as the baseline.
- Promote a variant only if it improves cross-class mAP or improves Car AP without hurting mAP on both KITTI detectors.
- Run all long training, purification, and detector evaluation jobs in `tmux`.
- Use both T4 GPUs whenever available: `CUDA_VISIBLE_DEVICES=0,1`, `NUM_GPUS=2`, `GPUS=0,1`.
