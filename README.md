# Subjective Quality Towards Simple yet Effective Geometry Processing for Solid G-PCC Application

This repository provides four implementations of the Geometry-based Solid Test Model (GeS-TM) v7.0-rc2, including three simple yet effective Wiener filter-based geometry post-processing methods designed to improve the subjective quality of compressed solid point clouds in G-PCC (Geometry-based Point Cloud Compression).

## Abstract

Geometry-based Point Cloud Compression (G-PCC) is a standard being developed by MPEG-3DGC (ISO/IEC SC29 WG7) for compressing dynamic point clouds. While the standard codec achieves competitive rate-distortion performance, the reconstructed point clouds often suffer from visual artifacts in geometry that degrade subjective quality. This work investigates simple yet effective Wiener filter-based geometry post-processing techniques for solid G-PCC content, including:

- **BWF (Bilateral Wiener Filter)**: Applies bilateral filtering principles within a Wiener filter framework to smooth geometry while preserving edges.
- **CIWF (Cross-component Independent Wiener Filter)**: Processes color components independently using Wiener filtering to reduce cross-component interference.
- **VCWF (Voxel-based Collaborative Wiener Filter)**: Leverages collaborative filtering across voxel neighborhoods, with separate processing for luma (Y) and chroma (Cb/Cr) components to better preserve perceptual features.

All three methods are implemented as extensions to the GeS-TM v7.0-rc2 reference software.

## Folder Structure

| Folder | Description |
|--------|-------------|
| `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2` | **Base** The original GES-TM v7rc2 source code used as the baseline. |
| `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-BWF` | The implementation of **BWF** in the paper |
| `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-CIWF` | The implementation of **CIWF** in the paper |
| `mpeg-pcc-ges-tm-ges-tm-v7.0-rc2-VCWF` | The implementation of **VCWF** in the paper |

Each folder contains a complete, self-contained implementation with the following internal structure:

```
├── cfg/                    # Configuration files for Common Test Conditions (CTCs)
│   ├── inter/              # Inter-frame prediction configurations
│   └── per-sequence/       # Per-sequence parameter configurations
├── dependencies/           # External dependencies (Eigen, etc.)
├── doc/                    # Documentation and manuals
├── global-motion-files/    # Global motion estimation files
├── scripts/                # Utility scripts (gen-cfg.sh, Makefile steps)
├── tmc3/                   # Core codec source code (C++)
│   ├── wiener_filter.cpp   # Wiener filter implementation (*WF variants only)
│   └── wiener_filter.h     # Wiener filter header (*WF variants only)
├── tools/                  # Auxiliary tools (ply-merge, etc.)
├── CMakeLists.txt          # CMake build configuration
├── COPYING                 # BSD License
└── README.md               # Original TMC13 README
```

### Key Differences Between Variants

The three Wiener filter variants differ primarily in:

1. **BWF**: The foundational Wiener filter implementation operating on reconstructed point cloud coordinates and colors. Uses a joint filtering approach with 3-channel coefficient storage (`coef_save[3][K_wiener]`).

2. **CIWF**: Extends BWF by independently optimizing filter coefficients per color component, reducing cross-component interference and improving chroma fidelity.

3. **VCWF**: Further extends the framework with collaborative voxel-based filtering. Separates luma (Y) and chroma (Cb/Cr) channels with dedicated coefficient buffers (`coef_save[2][K_wiener]` for chroma, `coef_Y_save[7][K_wiener]` for luma over 7 partitions), enabling finer control over perceptual quality.

## Requirements

- **CMake** >= 3.10
- **C++11** compatible compiler (GCC, Clang, MSVC 2017+)
- **Eigen** (included in `dependencies/`)
- **Operating Systems**: Linux, macOS, Windows

## Building

Each folder is built independently using CMake:

### Linux

```bash
cd mpeg-pcc-ges-tm-ges-tm-v7.0-rc2    # or -BWF, -CIWF, -VCWF
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### macOS

```bash
cd mpeg-pcc-ges-tm-ges-tm-v7.0-rc2
mkdir build && cd build
cmake .. -G Xcode
xcodebuild
```

### Windows

```powershell
cd mpeg-pcc-ges-tm-ges-tm-v7.0-rc2
md build && cd build
cmake .. -G "Visual Studio 17 2022 Win64"
# Open the generated Visual Studio solution and build
```

## Usage

### Generating Configuration Files

```bash
cd cfg
../scripts/gen-cfg.sh --all
```

### Encoding and Decoding

The `tmc3` binary functions as both encoder and decoder, selected via the `--mode` option:

```bash
# Encode
./build/tmc3/tmc3 --mode=encode -c cfg/encoder.cfg -o output.bin

# Decode
./build/tmc3/tmc3 --mode=decode -c cfg/decoder.cfg -o reconstructed.ply
```

For the Wiener filter variants (BWF, CIWF, VCWF), the filter is automatically applied during the decoding process. No additional command-line options are required.

### Automated Pipeline

The `scripts/Makefile.tmc13-step` provides a complete encode-decode-metric evaluation pipeline:

```bash
make -f scripts/Makefile.tmc13-step \
    -C experiment/output/dir/ \
    VPATH=cfg/path/ \
    ENCODER=build/tmc3/tmc3 \
    DECODER=build/tmc3/tmc3 \
    PCERROR=/path/to/pc_error \
    SRCSEQ=/path/to/input.ply
```

## License

This software is released under the **BSD 3-Clause License**. See the [COPYING](mpeg-pcc-ges-tm-ges-tm-v7.0-rc2/COPYING) file for details.

Copyright (c) 2017-2018, ISO/IEC. All rights reserved.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yuan2024subjective,
  title={Subjective Quality Towards Simple yet Effective Geometry Processing for Solid G-PCC Application},
  author={Yuan, Hui and others},
  journal={},
  year={2024}
}
```

## Contact

For questions and bug reports related to the base GeS-TM, please contact the MPEG 3DGC email reflector at <mpeg-3dgc@gti.ssr.upm.es> (registration required).

The authoritative location of the base reference software:
<https://git.mpeg.expert/MPEG/3dgh/g-pcc/software/tm/mpeg-pcc-ges-tm.git>
