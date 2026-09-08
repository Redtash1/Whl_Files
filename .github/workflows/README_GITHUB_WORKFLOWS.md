# GitHub Multi-Wheel Workflows v1

Copy the `.github` and `build_tools` folders into the ROOT of the GitHub wheel repository.

GitHub will keep every `.yml` file under `.github/workflows/` as a separate selectable workflow. This lets the repository become a wheel factory for Sage, GGUF, newer Torch/CUDA combinations, and Windows/Linux variants without replacing earlier builders.

## Included workflow

`Build SageAttention 2.2 SM75 - Py312 Torch2.6 CU126 Linux`

Locked compatibility target:

- Linux x86_64
- CPython 3.12
- Torch 2.6.0+cu126
- CUDA toolkit 12.6
- Turing / SM75

Expected wheel:

`sageattention-2.2.0+sm75.torch2.6cu126-cp312-cp312-linux_x86_64.whl`

## Build architecture

The workflow uses current SageAttention 2.2 package/API and native INT8 preprocessing, enables the required BLKK32 native preprocessing path, and replaces the qattn SM80 source pair with the proven Turing implementation from `Ph0rk0z/SageAttention2`. Public `sageattn()` dispatch is patched so SM75 uses the CUDA FP16-PV/FP32-accumulation path with the proven Turing geometry:

- BLKQ = 64
- WARPQ = 16
- BLKK = 32

Everything is compiled in one GitHub job against the SAME Python 3.12 / Torch 2.6.0+cu126 / CUDA 12.6 ABI.

UV provisions the Python environment. Native CUDA compilation is intentionally performed with:

`python -m pip wheel . --no-build-isolation --no-deps`

## Use

1. Extract the ZIP.
2. Copy `.github` and `build_tools` into the root of your GitHub repository.
3. Commit/push.
4. Open **Actions**.
5. Select **Build SageAttention 2.2 SM75 - Py312 Torch2.6 CU126 Linux**.
6. Click **Run workflow**.
7. Leave the default source refs for the first run.
8. Download the `sageattention-sm75-py312-torch260-cu126-linux` artifact.

The artifact contains the wheel, SHA256 checksum, and build manifest.

GitHub proves the wheel compiles, packages, imports in a CPU-hosted clean environment, and contains SM75 cubins. A short Kaggle T4 run is still required to prove actual SM75 runtime execution and performance.
