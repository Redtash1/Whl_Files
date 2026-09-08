#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, shutil, subprocess, sys
from pathlib import Path

TARGET_VERSION = "2.2.0+sm75.torch2.6cu126"
TARGET_WHEEL = f"sageattention-{TARGET_VERSION}-cp312-cp312-linux_x86_64.whl"


def run(cmd, cwd=None, env=None, check=True):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    cp = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(cp.stdout, flush=True)
    if check and cp.returncode:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(map(str, cmd))}")
    return cp


def replace_once(path: Path, old: str, new: str, label: str):
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label}: expected anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("PATCH:", label)


def patch_setup(setup: Path):
    text = setup.read_text(encoding="utf-8")

    # Explicit release version.
    text, n = re.subn(r"version\s*=\s*['\"]2\.2\.0['\"]",
                      f"version='{TARGET_VERSION}'", text, count=1)
    if n != 1:
        raise RuntimeError("setup.py version anchor not found")

    # Add SM75 state and supported architecture.
    if "HAS_SM75 = False" not in text:
        text, n = re.subn(r"(HAS_SM80\s*=\s*False\s*\n)", r"\1HAS_SM75 = False\n", text, count=1)
        if n != 1:
            raise RuntimeError("HAS_SM80 anchor not found")

    # Supported arch set exists in both older/current source families.
    m = re.search(r"SUPPORTED_ARCHS\s*=\s*\{([^}]*)\}", text)
    if m and '"7.5"' not in m.group(0) and "'7.5'" not in m.group(0):
        repl = m.group(0)[:-1].rstrip() + ', "7.5"}'
        text = text[:m.start()] + repl + text[m.end():]

    # Add the 7.5 branch immediately before the 8.0 branch.
    if 'capability.startswith("7.5")' not in text and "capability.startswith('7.5')" not in text:
        pat = r"(?P<i>[ \t]*)if capability\.startswith\([\"']8\.0[\"']\):\s*\n(?P=i)[ \t]+HAS_SM80\s*=\s*True\s*\n(?P=i)[ \t]+num\s*=\s*[\"']80[\"']"
        m = re.search(pat, text)
        if not m:
            raise RuntimeError("compute capability 8.0 branch not found")
        i = m.group("i")
        original = m.group(0)
        branch = (f'{i}if capability.startswith("7.5"):\n'
                  f'{i}    HAS_SM75 = True\n'
                  f'{i}    num = "75"\n'
                  f'{i}elif capability.startswith("8.0"):\n'
                  f'{i}    HAS_SM80 = True\n'
                  f'{i}    num = "80"')
        text = text[:m.start()] + branch + text[m.end():]

    # Build the qattn extension for SM75 too.
    text, n = re.subn(r"if HAS_SM80 or ", "if HAS_SM75 or HAS_SM80 or ", text, count=1)
    if n != 1:
        # Older source can have `if HAS_SM80:`.
        text, n = re.subn(r"if HAS_SM80\s*:", "if HAS_SM75 or HAS_SM80:", text, count=1)
    if n != 1:
        raise RuntimeError("qattn extension condition anchor not found")

    # Older v2.2 setup detects only physical GPUs. Replace that discovery with
    # TORCH_CUDA_ARCH_LIST support when needed, so GitHub CPU runners can cross-compile.
    if "Prefer TORCH_CUDA_ARCH_LIST" not in text and "arch_list_env = os.getenv(\"TORCH_CUDA_ARCH_LIST\"" not in text:
        start = text.find("compute_capabilities = set()")
        if start < 0:
            raise RuntimeError("compute_capabilities anchor not found")
        nvcc = text.find("nvcc_cuda_version = get_nvcc_cuda_version", start)
        if nvcc < 0:
            raise RuntimeError("nvcc version anchor not found")
        prefix = text[:start]
        suffix = text[nvcc:]
        discovery = '''compute_capabilities = set()\n# GitHub build: honor TORCH_CUDA_ARCH_LIST even without a physical GPU.\narch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()\nif arch_list_env:\n    for item in arch_list_env.replace(",", ";").split(";"):\n        it = item.strip().lower().replace("sm_", "").replace("compute_", "")\n        if not it:\n            continue\n        if it.endswith("+ptx"):\n            it = it[:-4]\n            suffix_ptx = "+PTX"\n        else:\n            suffix_ptx = ""\n        if len(it) == 2 and it.isdigit():\n            it = f"{it[0]}.{it[1]}"\n        compute_capabilities.add(it + suffix_ptx)\nif not compute_capabilities:\n    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0\n    for i in range(device_count):\n        major, minor = torch.cuda.get_device_capability(i)\n        compute_capabilities.add(f"{major}.{minor}")\n'''
        text = prefix + discovery + suffix

    setup.write_text(text, encoding="utf-8")
    print("PATCH: setup.py SM75 + GitHub cross-compile + explicit version")


def patch_blkk32(src: Path):
    candidates = [p for p in src.rglob("dispatch_utils.h") if "csrc" in str(p)]
    if not candidates:
        raise RuntimeError("dispatch_utils.h not found")
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "DISPATCH_BLOCK_SIZE" not in text:
            continue
        if re.search(r"case\s+32\s*:", text):
            print("PATCH: BLOCK_SIZE 32 already present", path)
            return
        # Duplicate the complete `case 64` arm through the next case/default.
        m = re.search(r"(?ms)^(?P<i>[ \t]*)case\s+64\s*:.*?(?=^(?P=i)(?:case\s+\d+\s*:|default\s*:))", text)
        if not m:
            continue
        block = m.group(0)
        block32 = re.sub(r"case\s+64\s*:", "case 32:", block, count=1)
        # The branch instantiates the block-size constant; convert only this cloned arm.
        block32 = re.sub(r"\b64\b", "32", block32)
        path.write_text(text[:m.start()] + block32 + text[m.start():], encoding="utf-8")
        print("PATCH: Sage2.2 native fused dispatcher accepts BLOCK_SIZE=32", path)
        return
    raise RuntimeError("could not safely patch BLOCK_SIZE=32")


def patch_core(core: Path):
    text = core.read_text(encoding="utf-8")

    # Public Sage dispatcher: T4/Turing -> CUDA FP16-PV path with FP32 accumulation.
    if 'if arch == "sm75"' not in text:
        anchor = 'if arch == "sm80":'
        if anchor not in text:
            raise RuntimeError("core.py sm80 dispatch anchor not found")
        route = ('if arch == "sm75":\n'
                 '        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout=tensor_layout, '
                 'is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, '
                 'qk_quant_gran="per_warp", pv_accum_dtype="fp32")\n'
                 '    elif arch == "sm80":')
        text = text.replace(anchor, route, 1)

    # Exact Turing Q/K preprocessing geometry proven on SM75:
    # BLKQ=64, WARPQ=16, BLKK=32. Keep upstream geometry for other architectures.
    old_pat = re.compile(
        r'q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda\('
        r'q, k, km, tensor_layout=tensor_layout, BLKQ=128, '
        r'WARPQ=\(16 if \(q\.size\(-1\) == 128 and pv_accum_dtype == "fp16\+fp32"\) else 32\), BLKK=64\)'
    )
    m = old_pat.search(text)
    if not m:
        # v2.2.0 source can use a simpler WARPQ=32 spelling.
        old_pat = re.compile(
            r'q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda\('
            r'q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64\)'
        )
        m = old_pat.search(text)
    if not m:
        raise RuntimeError("core.py per_warp Sage2.2 geometry anchor not found")

    replacement = '''if get_cuda_arch_versions()[q.device.index] == "sm75":\n            q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=32)\n        else:\n            q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64)'''
    text = text[:m.start()] + replacement + text[m.end():]
    core.write_text(text, encoding="utf-8")
    print("PATCH: core.py public SM75 dispatch + exact BLKQ64/WARPQ16/BLKK32 geometry")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--sage22-ref", required=True)
    ap.add_argument("--turing-ref", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    ws = a.workspace.resolve(); out = a.output.resolve()
    shutil.rmtree(ws, ignore_errors=True); ws.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({"CUDA_HOME": "/usr/local/cuda", "CUDACXX": "/usr/local/cuda/bin/nvcc",
                "TORCH_CUDA_ARCH_LIST": "7.5", "MAX_JOBS": env.get("MAX_JOBS", "2")})

    run([sys.executable, "-c",
         "import sys,torch;print(sys.version);print(torch.__version__);print(torch.version.cuda);"
         "assert sys.version_info[:2]==(3,12);assert torch.__version__.startswith('2.6.0');assert torch.version.cuda=='12.6'"], env=env)

    sage = ws / "SageAttention"
    turing = ws / "SageAttention2_Turing"
    run(["git", "clone", "--depth", "1", "--branch", a.sage22_ref,
         "https://github.com/thu-ml/SageAttention.git", sage])
    run(["git", "clone", "--depth", "1", "--branch", a.turing_ref,
         "https://github.com/Ph0rk0z/SageAttention2.git", turing])

    # Replace only the qattn kernel/binding with the proven Turing implementation.
    for rel in ("csrc/qattn/pybind_sm80.cpp", "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu"):
        src = turing / rel; dst = sage / rel
        if not src.exists() or not dst.exists():
            raise RuntimeError(f"qattn source path missing: {rel}")
        shutil.copy2(src, dst)
        print("TURING QATTN SOURCE:", rel)

    patch_setup(sage / "setup.py")
    patch_blkk32(sage)
    patch_core(sage / "sageattention" / "core.py")

    # Save patch evidence before compilation.
    run(["git", "diff", "--", "setup.py", "sageattention/core.py", "csrc"], cwd=sage, env=env)

    build_dist = ws / "dist"; build_dist.mkdir()
    run([sys.executable, "-m", "pip", "wheel", ".", "--no-build-isolation", "--no-deps", "-w", build_dist], cwd=sage, env=env)

    wheels = list(build_dist.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel, found {wheels}")
    wheel = wheels[0]
    if wheel.name != TARGET_WHEEL:
        raise RuntimeError(f"unexpected wheel filename: {wheel.name}; expected {TARGET_WHEEL}")
    shutil.copy2(wheel, out / wheel.name)
    print("\nFINAL:", out / wheel.name)

if __name__ == "__main__":
    main()
