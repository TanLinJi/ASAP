# Reference literature

This directory is for local reference papers and a committed literature index.

PDF files are intentionally kept local only and are ignored by Git. The committed content should be limited to bibliographic notes, links, and reading plans.

## Papers already placed locally

The following PDF files are present locally under `docs/references/`:

- **LiDAR-SPD: Improving Adversarial Robustness of 3D Object Detection via Spherical Projection and Diffusion**
  - Topic: LiDAR adversarial defense for 3D object detection.
  - Role in ASAP: inspiration and reported baseline, not code to reproduce directly.
- **Ada3Diff: Defending Against 3D Adversarial Point Clouds via Adaptive Diffusion**
  - Topic: adaptive diffusion purification for adversarial point clouds.
  - Role in ASAP: diffusion purification reference.
- **Diffusion Models-Based Purification for Common Corruptions on Robust 3D Object Detection**
  - Topic: diffusion purification for robust LiDAR 3D object detection under common corruptions.
  - Role in ASAP: robustness and LiDAR purification reference.

## Recommended papers to download next

### Core adversarial attack and defense references

- **PointDP: Diffusion-driven Purification against Adversarial Attacks on 3D Point Cloud Recognition**
  - Venue/year: International Conference on Machine Learning, 2023.
  - Link: https://arxiv.org/abs/2208.09801
  - Why download: key diffusion purification baseline for 3D point clouds.
- **ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**
  - Year: 2023 preprint.
  - Link: https://arxiv.org/abs/2312.03085
  - Why download: relevant LiDAR object detection defense baseline.
- **A Comprehensive Study of the Robustness for LiDAR-based 3D Object Detection**
  - Venue/year: International Journal of Computer Vision, 2023.
  - Link: https://arxiv.org/abs/2212.10230
  - Why download: broad robustness benchmark and threat model reference.
- **Towards Robust LiDAR-based Perception in Autonomous Driving: General Black-box Adversarial Sensor Attack and Countermeasures**
  - Venue/year: USENIX Security, 2020.
  - Link: https://www.usenix.org/conference/usenixsecurity20/presentation/sun
  - Why download: important LiDAR sensor attack baseline related to point injection and spoofing.
- **Adversarial Point Cloud Perturbations against 3D Object Detection in Autonomous Driving Systems**
  - Year: 2021.
  - Link: https://arxiv.org/search/?query=Adversarial+Point+Cloud+Perturbations+against+3D+Object+Detection+in+Autonomous+Driving+Systems&searchtype=all
  - Why download: important point perturbation attack baseline for 3D object detection.

### Recent related papers from top venues

- **A New Adversarial Perspective for LiDAR-based 3D Object Detection**
  - Venue/year: AAAI Conference on Artificial Intelligence, 2025.
  - Link: https://ojs.aaai.org/index.php/AAAI/article/view/33152
  - Why download: recent LiDAR adversarial object perturbation work.
- **DiffuBox: Refining 3D Object Detection with Point Diffusion**
  - Venue/year: Neural Information Processing Systems, 2024.
  - Link: https://proceedings.neurips.cc/paper_files/paper/2024/hash/bbd35a696d85afab1249423dbd6e1041-Abstract-Conference.html
  - Why download: recent point diffusion method for 3D detection refinement.
- **CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation**
  - Venue/year: European Conference on Computer Vision, 2024.
  - Link: https://arxiv.org/abs/2407.16193
  - Why download: diffusion-guided point cloud input adaptation, useful for positioning ASAP as detector-agnostic input purification.
- **PCoTTA: Continual Test-Time Adaptation for Multi-Task Point Cloud Understanding**
  - Venue/year: Neural Information Processing Systems, 2024.
  - Link: https://arxiv.org/abs/2411.00632
  - Why download: recent point cloud test-time adaptation reference.
- **Test-Time Adaptation in Point Clouds: Leveraging Sampling Variation with Weight Averaging**
  - Venue/year: Winter Conference on Applications of Computer Vision, 2025.
  - Link: https://openaccess.thecvf.com/content/WACV2025/html/Bahri_Test-Time_Adaptation_in_Point_Clouds_Leveraging_Sampling_Variation_with_Weight_WACV_2025_paper.html
  - Why download: recent point cloud test-time adaptation baseline.

### Robust LiDAR perception under weather and corruptions

- **Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection**
  - Venue/year: AAAI Conference on Artificial Intelligence, 2024.
  - Link: https://ojs.aaai.org/index.php/AAAI/search/search?query=Sunshine%20to%20Rainstorm%20Cross-Weather%20Knowledge%20Distillation%20for%20Robust%203D%20Object%20Detection
  - Why download: robust LiDAR detection under weather shift.
- **LiDAR-based All-weather 3D Object Detection via Prompting and Distilling 4D Radar**
  - Venue/year: European Conference on Computer Vision, 2024.
  - Link: https://eccv.ecva.net/virtual/2024/poster/1725
  - Why download: all-weather robust LiDAR detection reference.

## Reading priority

1. LiDAR-SPD, Ada3Diff, PointDP.
2. Sun20-style attack and Wang21-style perturbation attack papers.
3. ScAR and the IJCV robustness study.
4. DiffuBox and CloudFixer for recent diffusion-based 3D point cloud context.
5. Weather/corruption robustness papers for broader related work.

## Abbreviation notes

- **ASAP** means Adaptive Spherical Anomaly-Guided Purification.
- **LiDAR** means Light Detection and Ranging.
- **SPU** means Spherical Purification Unit.
- **ASR** means Attack Success Rate.
- **mAP** means mean Average Precision.
- **TTA** means Test-Time Adaptation.
- **NeurIPS** means Neural Information Processing Systems.
- **ECCV** means European Conference on Computer Vision.
- **AAAI** means Association for the Advancement of Artificial Intelligence.
- **IJCV** means International Journal of Computer Vision.
- **WACV** means Winter Conference on Applications of Computer Vision.
