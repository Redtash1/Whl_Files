from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SETUP = ROOT / "setup.py"
MMA = ROOT / "csrc" / "mma.cuh"
VERSION = "2.2.0+sm75.torch2.12cu130.turingmma1"

def replace_function(text: str, name: str, replacement: str) -> str:
    pos = text.find(name)
    if pos < 0:
        raise RuntimeError(f"Function anchor not found: {name}")
    brace = text.find("{", pos)
    if brace < 0:
        raise RuntimeError(f"Opening brace not found: {name}")
    depth = 0
    i = brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                start = text.rfind("\n", 0, pos) + 1
                return text[:start] + replacement.rstrip() + "\n" + text[i+1:]
        i += 1
    raise RuntimeError(f"Closing brace not found: {name}")

def patch_setup():
    s = SETUP.read_text(encoding="utf-8")
    if "HAS_SM75 = False" not in s:
        s = s.replace("    HAS_SM80 = False", "    HAS_SM75 = False\n    HAS_SM80 = False", 1)

    s = s.replace(
        'SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0", "10.0", "12.0", "12.1"}',
        'SUPPORTED_ARCHS = {"7.5", "8.0", "8.6", "8.9", "9.0", "10.0", "12.0", "12.1"}'
    )

    s = s.replace(
        '        if major < 8:\n            warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")\n            continue',
        '        if major < 7 or (major == 7 and minor < 5):\n            warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")\n            continue'
    )

    old = '        if capability.startswith("8.0"):\n            HAS_SM80 = True\n            num = "80"'
    new = '        if capability.startswith("7.5"):\n            HAS_SM75 = True\n            num = "75"\n        elif capability.startswith("8.0"):\n            HAS_SM80 = True\n            num = "80"'
    if old in s:
        s = s.replace(old, new, 1)
    elif 'capability.startswith("7.5")' not in s:
        raise RuntimeError("setup.py architecture dispatch anchor changed")

    s = s.replace(
        "if HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM100 or HAS_SM120 or HAS_SM121:",
        "if HAS_SM75 or HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM100 or HAS_SM120 or HAS_SM121:",
        1
    )

    s = re.sub(r"version\s*=\s*['\"]2\.2\.0['\"]", f"version='{VERSION}'", s, count=1)
    SETUP.write_text(s, encoding="utf-8")

def patch_mma():
    s = MMA.read_text(encoding="utf-8")

    f32 = r'''
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                      uint32_t* B) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  if constexpr (mma_mode == MMAMode::kInit) C[0] = C[1] = C[2] = C[3] = 0.0f;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3},{%4, %5, %6, %7},{%8, %9},{%0, %1, %2, %3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
#elif (__CUDA_ARCH__ >= 750)
  if constexpr (mma_mode == MMAMode::kInit) C[0] = C[1] = C[2] = C[3] = 0.0f;
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3},{%4, %5},{%6},{%0, %1, %2, %3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]));
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3},{%4, %5},{%6},{%0, %1, %2, %3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[2]), "r"(A[3]), "r"(B[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}
'''

    f16 = r'''
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                      uint32_t* B) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  if constexpr (mma_mode == MMAMode::kInit) C[0] = C[1] = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1},{%2, %3, %4, %5},{%6, %7},{%0, %1};\n"
      : "+r"(C[0]), "+r"(C[1])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
#elif (__CUDA_ARCH__ >= 750)
  if constexpr (mma_mode == MMAMode::kInit) C[0] = C[1] = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0, %1},{%2, %3},{%4},{%0, %1};\n"
      : "+r"(C[0]), "+r"(C[1])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]));
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0, %1},{%2, %3},{%4},{%0, %1};\n"
      : "+r"(C[0]), "+r"(C[1])
      : "r"(A[2]), "r"(A[3]), "r"(B[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}
'''

    i8 = r'''
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                    uint32_t* B) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  if constexpr (mma_mode == MMAMode::kInit) C[0] = C[1] = C[2] = C[3] = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3},{%4, %5, %6, %7},{%8, %9},{%0, %1, %2, %3};\n"
      : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
#elif (__CUDA_ARCH__ >= 750)
  if constexpr (mma_mode == MMAMode::kInit) C[0] = C[1] = C[2] = C[3] = 0;
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
      : "+r"(C[0]), "+r"(C[1]) : "r"(A[0]), "r"(B[0]));
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
      : "+r"(C[2]), "+r"(C[3]) : "r"(A[1]), "r"(B[0]));
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
      : "+r"(C[0]), "+r"(C[1]) : "r"(A[2]), "r"(B[1]));
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
      : "+r"(C[2]), "+r"(C[3]) : "r"(A[3]), "r"(B[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}
'''
    s = replace_function(s, "mma_sync_m16n8k16_row_col_f16f16f32", f32)
    s = replace_function(s, "mma_sync_m16n8k16_row_col_f16f16f16", f16)
    s = replace_function(s, "mma_sync_m16n8k32_row_col_s8s8s32", i8)
    MMA.write_text(s, encoding="utf-8")

def sanity():
    setup = SETUP.read_text(encoding="utf-8")
    mma = MMA.read_text(encoding="utf-8")
    for token in ["HAS_SM75 = False", '"7.5"', 'num = "75"', VERSION, "if HAS_SM75 or HAS_SM80"]:
        if token not in setup:
            raise RuntimeError(f"setup sanity missing: {token}")
    for token in [
        "m16n8k8.row.col.f32.f16.f16.f32",
        "m16n8k8.row.col.f16.f16.f16.f16",
        "m8n8k16.row.col.s32.s8.s8.s32",
    ]:
        if token not in mma:
            raise RuntimeError(f"mma sanity missing: {token}")
    print("[PASS] SM75 setup.py architecture route installed")
    print("[PASS] Turing FP16 MMA compatibility installed")
    print("[PASS] Turing INT8 MMA compatibility installed")
    print("[PASS] Wheel version:", VERSION)

patch_setup()
patch_mma()
sanity()
