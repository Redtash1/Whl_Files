from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SETUP = ROOT / "setup.py"
MMA = ROOT / "csrc" / "mma.cuh"
VERSION = "2.2.0+sm75.torch2.12cu130"

def replace_function(text: str, name: str, replacement: str) -> str:
    pos = text.find(name)
    if pos < 0:
        raise RuntimeError(f"Function anchor not found: {name}")
    brace = text.find("{", pos)
    if brace < 0:
        raise RuntimeError(f"Opening brace not found: {name}")

    # Include the function's template declaration in the replacement range.
    # v1.1 started at the function declaration and left the original
    # `template <MMAMode ...>` line behind, producing duplicate template
    # clauses when the replacement (which also contains a template line)
    # was inserted.
    decl_start = text.rfind("\n", 0, pos) + 1
    previous_line_end = max(0, decl_start - 1)
    previous_line_start = text.rfind("\n", 0, previous_line_end) + 1
    previous_line = text[previous_line_start:previous_line_end].strip()
    start = previous_line_start if previous_line.startswith("template <") else decl_start

    depth = 0
    i = brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
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

    # The SageAttention qattn kernel does NOT call the m16n8 helpers directly.
    # compute_int_qk() calls m16n16k32, and compute_fp16_sv*() calls m16n16k16.
    # v1.2 left those wrappers Ampere-only, so SM75 hit RUNTIME_ASSERT/__brkpt()
    # before our Turing m16n8 MMA code could execute.
    f32_n16 = r"""
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                       uint32_t* B) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  mma_sync_m16n8k16_row_col_f16f16f32<mma_mode>(C, A, B);
  mma_sync_m16n8k16_row_col_f16f16f32<mma_mode>(C + 4, A, B + 2);
#elif (__CUDA_ARCH__ >= 750)
  mma_sync_m16n8k16_row_col_f16f16f32<mma_mode>(C, A, B);
  mma_sync_m16n8k16_row_col_f16f16f32<mma_mode>(C + 4, A, B + 2);
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}
"""

    f16_n16 = r"""
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                       uint32_t* B) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  mma_sync_m16n8k16_row_col_f16f16f16<mma_mode>(C, A, B);
  mma_sync_m16n8k16_row_col_f16f16f16<mma_mode>(C + 2, A, B + 2);
#elif (__CUDA_ARCH__ >= 750)
  mma_sync_m16n8k16_row_col_f16f16f16<mma_mode>(C, A, B);
  mma_sync_m16n8k16_row_col_f16f16f16<mma_mode>(C + 2, A, B + 2);
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}
"""

    i8_n16 = r"""
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  mma_sync_m16n8k32_row_col_s8s8s32<mma_mode>(C, A, B);
  mma_sync_m16n8k32_row_col_s8s8s32<mma_mode>(C + 4, A, B + 2);
#elif (__CUDA_ARCH__ >= 750)
  mma_sync_m16n8k32_row_col_s8s8s32<mma_mode>(C, A, B);
  mma_sync_m16n8k32_row_col_s8s8s32<mma_mode>(C + 4, A, B + 2);
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}
"""

    # Tensor-core denominator accumulation also remained Ampere-only in v1.2.
    # For SM75, reproduce the CUDA-core partial sums and perform the same
    # 4-lane row reduction immediately, because the caller's TensorCore mode
    # does not perform the later normalize_d() shuffle.
    rowsum = r"""
__device__ __forceinline__ void rowsum_f16f16f32(float* d, uint32_t* s) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  asm volatile(
      "{\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, _, %1, _},"
      "{%2, %3, %4, %5},"
      "{%6, %7},"
      "{%8, 0., %9, 0.};\n"
      "}\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]),
        "r"(1006648320), "r"(1006648320), "f"(d[0]), "f"(d[1]));
#elif (__CUDA_ARCH__ >= 750)
  const half2 h0 = *reinterpret_cast<const half2*>(&s[0]);
  const half2 h1 = *reinterpret_cast<const half2*>(&s[1]);
  const half2 h2 = *reinterpret_cast<const half2*>(&s[2]);
  const half2 h3 = *reinterpret_cast<const half2*>(&s[3]);

  const float2 f0 = __half22float2(h0);
  const float2 f1 = __half22float2(h1);
  const float2 f2 = __half22float2(h2);
  const float2 f3 = __half22float2(h3);

  float r0 = f0.x + f0.y + f2.x + f2.y;
  float r1 = f1.x + f1.y + f3.x + f3.y;

  r0 += __shfl_xor_sync(0xffffffff, r0, 0x1);
  r0 += __shfl_xor_sync(0xffffffff, r0, 0x2);
  r1 += __shfl_xor_sync(0xffffffff, r1, 0x1);
  r1 += __shfl_xor_sync(0xffffffff, r1, 0x2);

  d[0] += r0;
  d[1] += r1;
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for rowsum");
#endif
}
"""

    s = replace_function(s, "mma_sync_m16n8k16_row_col_f16f16f32", f32)
    s = replace_function(s, "mma_sync_m16n8k16_row_col_f16f16f16", f16)
    s = replace_function(s, "mma_sync_m16n8k32_row_col_s8s8s32", i8)

    # These are the wrappers actually used by SageAttention's SM80 qattn kernel.
    s = replace_function(s, "mma_sync_m16n16k16_row_col_f16f16f32", f32_n16)
    s = replace_function(s, "mma_sync_m16n16k16_row_col_f16f16f16", f16_n16)
    s = replace_function(s, "mma_sync_m16n16k32_row_col_s8s8s32", i8_n16)
    s = replace_function(s, "rowsum_f16f16f32", rowsum)

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

    # Ensure we did not leave the original template declaration in place.
    if re.search(r"template\s*<MMAMode[^\n]*>\s*\n\s*template\s*<MMAMode", mma):
        raise RuntimeError("duplicate MMAMode template declaration detected")

    for token in [
        "mma_sync_m16n16k32_row_col_s8s8s32",
        "mma_sync_m16n16k16_row_col_f16f16f32",
        "mma_sync_m16n16k16_row_col_f16f16f16",
        "Unsupported CUDA architecture for rowsum",
        "__shfl_xor_sync(0xffffffff, r0, 0x1)",
    ]:
        if token not in mma:
            raise RuntimeError(f"SM75 live-qattn sanity missing: {token}")

    print("[PASS] No duplicate template declarations")
    print("[PASS] SM75 setup.py architecture route installed")
    print("[PASS] Turing m16n8 FP16 MMA compatibility installed")
    print("[PASS] Turing m16n8 INT8 MMA compatibility installed")
    print("[PASS] Turing m16n16 QK/SV wrappers installed")
    print("[PASS] Turing denominator rowsum fallback installed")
    print("[PASS] Wheel version:", VERSION)

patch_setup()
patch_mma()
sanity()
