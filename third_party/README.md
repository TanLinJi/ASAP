# third_party

This directory hosts external code that ASAP depends on. ASAP itself stays a single, independent Git repository, so any external project here must follow strict rules.

## Rules

- **Always re-clone from upstream**, do not copy from any pre-existing local clone on the machine.
- **Remove the inner `.git` directory** of the cloned project. This avoids nesting another Git repository inside ASAP and keeps `git status` clean.
- **Keep the upstream `LICENSE`** and any `NOTICE` or copyright files inside the cloned project.
- **Record provenance** for each project in this README, including:
  - upstream URL
  - commit hash or release tag
  - license name
  - reason ASAP depends on it
  - list of any local modifications, if any
- **Do not commit large binaries** such as datasets, checkpoints, or logs that ship with the upstream project.
- **Do not change upstream license files**.

## Default ignore rule

The ASAP `.gitignore` ignores the contents of `third_party/` by default, except for `third_party/README.md` and `third_party/.gitkeep`. If we want to commit specific upstream source files, we either:

- selectively un-ignore them with explicit `!third_party/<project>/<path>` rules, or
- store the upstream source locally only and treat it as a runtime dependency that the user re-clones with a documented command.

For a typical research workflow we prefer the second option, so this repository remains small and clean.

## Recommended future entries

The following projects may be vendored later under this directory using the rules above:

- OpenPCDet, used as the LiDAR detection backend for clean and attacked evaluation.
- PointDP-style diffusion reference, used as the basis of the diffusion purifier.
- Selected attack implementations for KITTI LiDAR scenes.

## Vendoring template

Replace `<project_name>` and the upstream URL as needed. Run from this `third_party` directory.

```bash
git clone <upstream_url> <project_name>
cd <project_name>
git log -1 --pretty=format:'%H'  # record this commit hash in this README
cd ..
rm -rf <project_name>/.git
```

Then add an entry under "Vendored projects" below.

## Vendored projects

### OpenPCDet

- **Upstream URL**: https://github.com/open-mmlab/OpenPCDet
- **Recorded commit**: TODO: fill in from upstream before removing `.git` on a fresh clone
- **License**: Apache License 2.0
- **Purpose**: LiDAR 3D object detection backend for KITTI PointPillars baselines and detector evaluation.
- **Local modifications**:
  - `pcdet/datasets/__init__.py`: make `Argo2Dataset` registration optional when the Argoverse 2 dependency `av2` is not installed, so KITTI-only data preparation can run without installing unrelated Argoverse 2 packages.
  - `data/kitti`: replaced OpenPCDet's template KITTI directory with a symlink to `/root/autodl-tmp/ASAP/data/kitti`. The original template directory is kept locally as `data/kitti_openpcdet_template`.

