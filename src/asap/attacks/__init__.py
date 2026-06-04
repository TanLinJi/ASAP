"""ASAP adversarial attack implementations for LiDAR point clouds.

Three canonical attack families:
- point_injection: Insert adversarial point clusters near target objects.
- point_perturbation: Perturb existing object points along adversarial directions.
- point_dropping: Remove a fraction of object surface points.

Each attack reads clean KITTI/nuScenes velodyne bins and GT boxes,
produces attacked velodyne bins + a JSONL metadata log.
"""
