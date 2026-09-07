# MiniMax-H3 T4 GGUF CUDA GitHub Builder

This package builds the Linux Kaggle T4 version of `llamacpp_gguf_cuda`
without consuming Kaggle GPU time.

## Target

- Linux x86_64
- Python 3.12
- Torch 2.6.0+cu126
- CUDA toolkit 12.6
- Tesla T4 / SM75
- `TORCH_CUDA_ARCH_LIST=7.5`

## Use

Copy `.github/workflows/build_t4_gguf_cuda.yml` into a GitHub repository.

Then open:

**GitHub → Actions → Build MiniMax-H3 T4 GGUF CUDA Wheel → Run workflow**

When it succeeds, the wheel is available in two places:

1. The workflow's **Artifacts** section.
2. A GitHub Release named `minimax-h3-t4-gguf-cu126`.

Use the Release `.whl` URL in the Kaggle notebook so future runs skip compilation.

## Important

The workflow compiles the current `deepbeepmeep/kernels` source. If you need
byte-for-byte behavior matching a specific older/custom Windows wheel such as
`1.0.2+torch260cu126py312.custom10`, the matching source commit/custom patch set
is required. The filename alone is not enough to reconstruct a custom build.
