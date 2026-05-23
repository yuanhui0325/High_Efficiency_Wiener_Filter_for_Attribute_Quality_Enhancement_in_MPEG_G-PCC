# High Efficiency Wiener Filter for Attribute Quality Enhancement in MPEG G-PCC

This repository provides the source code used in the paper
**High Efficiency Wiener Filter for Attribute Quality Enhancement in
MPEG G-PCC**.

## Repository Structure

The repository is built around the MPEG G-PCC test model and includes four
implementations corresponding to different Wiener filter configurations.

| Path | Description |
| --- | --- |
| `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/` | The original GES-TM v7.0-rc2 source code used as the baseline. |
| `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-BWF/` | The implementation of **Bilateral Wiener Filter (BWF)** in the paper. |
| `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-CIWF/` | The implementation of **Cross-component Independent Wiener Filter (CIWF)** in the paper. |
| `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-VCWF/` | The implementation of **Voxel-based Collaborative Wiener Filter (VCWF)** in the paper. |

## Building

Each code project contains its own README file with build and usage
instructions:

- Baseline GES-TM v7.0-rc2:
  [`mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/README.md`](mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/README.md)
- Bilateral Wiener Filter (BWF):
  [`mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-BWF/mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/README.md`](mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-BWF/mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/README.md)
- Cross-component Independent Wiener Filter (CIWF):
  [`mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-CIWF/mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/README.md`](mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-CIWF/mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/README.md)
- Voxel-based Collaborative Wiener Filter (VCWF):
  [`mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-VCWF/mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/README.md`](mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-VCWF/mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/README.md)

All four projects are based on CMake. A typical build workflow is:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Please refer to the README file in each project directory for platform-specific
commands, release build options, and usage examples.

## Notes

- `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/` should be treated as the reference
  GES-TM v7.0-rc2 code.
- `BWF`, `CIWF`, and `VCWF` are modified versions of the baseline code for the
  three Wiener filter methods proposed in the paper.
