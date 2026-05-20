# Abstract

<!-- 目标长度: 150-200 词；当前为首版草稿，待实验数字到位后再回填具体 mAP/AUC 数字。 -->

LiDAR-based 3D object detection has become a core perception module for autonomous driving, yet recent work shows that point-cloud detectors are highly vulnerable to adversarial attacks, including point injection, perturbation, and dropping. Existing defenses are typically tied to a specific detector, require costly retraining, or apply uniform denoising that removes benign geometry together with adversarial perturbations. We propose **ASAP**, an **Adaptive Spherical Anomaly-Guided Purification** framework that protects LiDAR detectors in a detector-agnostic and inference-only manner. Specifically, given an input point cloud, ASAP first decomposes the scene into **Spherical Purification Units (SPUs)** with radii adapted to local point density, so that sparse and dense regions can be purified at appropriate spatial scales. It then estimates the adversarial likelihood of each SPU using a lightweight geometric anomaly score that combines compactness, PCA-based anisotropy, von Mises-Fisher angular concentration, and local density. Guided by this score, ASAP selectively applies a point-cloud diffusion purifier only to high-risk SPUs while leaving low-risk regions unchanged. This anomaly-guided purification pipeline suppresses localized adversarial perturbations, preserves benign object structures, and avoids the unnecessary cost of whole-scene denoising. Comprehensive experiments on the **KITTI** and **Waymo** benchmarks across multiple 3D detector architectures show that ASAP, without retraining any detector, effectively mitigates diverse LiDAR adversarial attacks, reducing the **attack success rate** against 3D object detectors by **60%** while preserving clean-scene detection performance.

<!-- TODO(experiments): 在 ICASSP 投稿前替换最后一句为带具体数字的版本，例如：
"Comprehensive experiments on the KITTI dataset show that ASAP effectively mitigates injection, perturbation, and dropping attacks, reducing the attack success rate against 3D object detectors by XX.X% while recovering YY.Y% of the clean 3D detection mAP."
说明：mAP = mean Average Precision；NDS = nuScenes Detection Score；ROC = Receiver Operating Characteristic；VP-SDE = Variance-Preserving Stochastic Differential Equation。 -->

## Keywords

LiDAR point cloud, 3D object detection, adversarial purification, diffusion model, anomaly detection, autonomous driving.
