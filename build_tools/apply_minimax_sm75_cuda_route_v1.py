from pathlib import Path
import ast, shutil, time, re

TARGET = Path("/kaggle/working/minimax-h3/shared/sage2_core.py")
if not TARGET.exists():
    raise FileNotFoundError(TARGET)

src = TARGET.read_text(encoding="utf-8")
MARK = "# H3_T4_SAGE2_CUDA_SM75_ROUTE_V1"
if MARK in src:
    print("[PASS] T4 Sage2 CUDA route already installed.")
    raise SystemExit(0)

pattern = re.compile(r'(?ms)^(\s*)if arch == "sm75":\n.*?(?=^\1elif arch == "sm80":)')
m = pattern.search(src)
if not m:
    raise RuntimeError("Existing SM75 Sage2 branch not found. No source changed.")

indent = m.group(1)
replacement = (
    f'{indent}if arch == "sm75":\n'
    f'{indent}    {MARK}\n'
    f'{indent}    if not getattr(sageattn, "_h3_sm75_cuda_notice", False):\n'
    f'{indent}        print("[SageAttention] Tesla T4/SM75: SageAttention 2.2 compiled Turing MMA CUDA path")\n'
    f'{indent}        sageattn._h3_sm75_cuda_notice = True\n'
    f'{indent}    return sageattn_qk_int8_pv_fp16_cuda(\n'
    f'{indent}        qkv_list,\n'
    f'{indent}        tensor_layout=tensor_layout,\n'
    f'{indent}        is_causal=is_causal,\n'
    f'{indent}        sm_scale=sm_scale,\n'
    f'{indent}        return_lse=return_lse,\n'
    f'{indent}        pv_accum_dtype="fp32",\n'
    f'{indent}    )\n'
)
patched = src[:m.start()] + replacement + src[m.end():]
ast.parse(patched, filename=str(TARGET))
compile(patched, str(TARGET), "exec")

backup = TARGET.with_suffix(f".py.pre_sm75_cuda_route_{int(time.time())}")
shutil.copy2(TARGET, backup)
TARGET.write_text(patched, encoding="utf-8")
print("[PASS] MiniMax-H3 SM75 Sage2 route changed: Triton -> compiled CUDA")
print("[PASS] AST + compile validation passed")
print("[PASS] Backup:", backup)
