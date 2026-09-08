#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import py_compile
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

TARGET_VERSION = "2.2.0+sm75.torch2.6cu126"
TARGET_WHEEL = f"sageattention-{TARGET_VERSION}-cp312-cp312-linux_x86_64.whl"


def run(cmd, cwd=None, env=None, check=True):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    cp = subprocess.run(
        list(map(str, cmd)),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(cp.stdout, flush=True)
    if check and cp.returncode:
        raise RuntimeError(
            f"command failed ({cp.returncode}): {' '.join(map(str, cmd))}"
        )
    return cp


def replace_once(text: str, pattern: str, replacement, label: str, flags=0) -> str:
    out, n = re.subn(pattern, replacement, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {n}")
    return out


def syntax_check(path: Path):
    py_compile.compile(str(path), doraise=True)
    print("PREFLIGHT: Python syntax OK:", path)


def patch_turing_setup(setup: Path):
    text = setup.read_text(encoding="utf-8")

    if "HAS_SM75 = False" not in text:
        raise RuntimeError("Turing setup.py does not contain HAS_SM75 support")
    if '"7.5"' not in text and "'7.5'" not in text:
        raise RuntimeError("Turing setup.py does not list SM75 as supported")

    start = text.find("compute_capabilities = set()")
    stop = text.find("nvcc_cuda_version = get_nvcc_cuda_version", start)
    if start < 0 or stop < 0:
        raise RuntimeError("Turing compute-capability discovery anchors not found")

    discovery = '''compute_capabilities = set()
# GitHub CPU runner: honor TORCH_CUDA_ARCH_LIST for cross-compilation.
arch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
if arch_list_env:
    for item in arch_list_env.replace(",", ";").split(";"):
        it = item.strip().lower().replace("sm_", "").replace("compute_", "")
        if not it:
            continue
        if it.endswith("+ptx"):
            it = it[:-4]
            suffix_ptx = "+PTX"
        else:
            suffix_ptx = ""
        if len(it) == 2 and it.isdigit():
            it = f"{it[0]}.{it[1]}"
        compute_capabilities.add(it + suffix_ptx)

if not compute_capabilities:
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            warnings.warn(
                f"skipping GPU {i} with compute capability {major}.{minor}"
            )
            continue
        compute_capabilities.add(f"{major}.{minor}")

'''
    text = text[:start] + discovery + text[stop:]
    setup.write_text(text, encoding="utf-8")
    syntax_check(setup)

    verify = setup.read_text(encoding="utf-8")
    required = (
        "HAS_SM75 = False",
        "TORCH_CUDA_ARCH_LIST",
        'name="sageattention._qattn_sm80"',
    )
    for item in required:
        if item not in verify:
            raise RuntimeError(f"Turing setup verification missing: {item}")
    if not re.search(r'capability\.startswith\(["\']7\.5["\']\)', verify):
        raise RuntimeError("Turing setup verification missing 7.5 capability branch")
    print("PATCH: Turing setup supports GitHub SM75 cross-compile")


def patch_sage22_setup(setup: Path):
    text = setup.read_text(encoding="utf-8")

    text = replace_once(
        text,
        r"(?m)^(\s*)version\s*=\s*['\"]2\.2\.0['\"]\s*,?\s*$",
        lambda m: f"{m.group(1)}version='{TARGET_VERSION}',",
        "Sage2.2 version",
    )

    if "TORCH_CUDA_ARCH_LIST" not in text:
        raise RuntimeError(
            "Official Sage2.2 setup.py no longer contains TORCH_CUDA_ARCH_LIST support"
        )

    if not re.search(r"(?m)^\s*HAS_SM75\s*=\s*False\s*$", text):
        pat = re.compile(r"(?m)^(?P<i>[ \t]*)HAS_SM80\s*=\s*False\s*$")
        m = pat.search(text)
        if not m:
            raise RuntimeError("HAS_SM80 declaration anchor not found")
        insertion = f"{m.group('i')}HAS_SM75 = False\n{m.group(0)}"
        text = text[:m.start()] + insertion + text[m.end():]

    m = re.search(
        r"(?m)^(?P<i>[ \t]*)SUPPORTED_ARCHS\s*=\s*\{(?P<body>[^}]*)\}\s*$",
        text,
    )
    if not m:
        raise RuntimeError("SUPPORTED_ARCHS anchor not found")
    if '"7.5"' not in m.group(0) and "'7.5'" not in m.group(0):
        body = m.group("body").rstrip()
        if body and not body.endswith(","):
            body += ","
        new_line = f'{m.group("i")}SUPPORTED_ARCHS = {{{body} "7.5"}}'
        text = text[:m.start()] + new_line + text[m.end():]

    if not re.search(r'capability\.startswith\(["\']7\.5["\']\)', text):
        pat = re.compile(
            r'(?m)^(?P<i>[ \t]*)if capability\.startswith\(["\']8\.0["\']\):\s*\n'
            r'(?P=i)(?P<b>[ \t]+)HAS_SM80\s*=\s*True\s*\n'
            r'(?P=i)(?P=b)num\s*=\s*["\']80["\']\s*$'
        )
        m = pat.search(text)
        if not m:
            raise RuntimeError("8.0 compute-capability branch anchor not found")
        i, b = m.group("i"), m.group("b")
        branch = (
            f'{i}if capability.startswith("7.5"):\n'
            f'{i}{b}HAS_SM75 = True\n'
            f'{i}{b}num = "75"\n'
            f'{i}elif capability.startswith("8.0"):\n'
            f'{i}{b}HAS_SM80 = True\n'
            f'{i}{b}num = "80"'
        )
        text = text[:m.start()] + branch + text[m.end():]

    if re.search(r"if\s+HAS_SM75\s+or\s+HAS_SM80", text):
        raise RuntimeError(
            "Unsafe Sage2.2 setup state: stock qattn condition includes HAS_SM75"
        )

    setup.write_text(text, encoding="utf-8")
    syntax_check(setup)

    verify = setup.read_text(encoding="utf-8")
    checks = {
        "indented HAS_SM75": bool(re.search(r"(?m)^[ \t]+HAS_SM75 = False$", verify)),
        "supported 7.5": ('"7.5"' in verify or "'7.5'" in verify),
        "SM75 branch": bool(re.search(r'capability\.startswith\(["\']7\.5["\']\)', verify)),
        "target version": f"version='{TARGET_VERSION}'" in verify,
        "stock qattn remains SM80+": "if HAS_SM80 or HAS_SM86" in verify,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(f"Sage2.2 setup verification failed: {bad}")

    print("PATCH: Sage2.2 setup SM75 fused preprocessing + explicit version")


def patch_blkk32(src: Path):
    path = src / "csrc" / "dispatch_utils.h"
    if not path.is_file():
        raise RuntimeError(f"Expected Sage2.2 dispatch header missing: {path}")

    text = path.read_text(encoding="utf-8")
    macro_start = text.find("#define DISPATCH_BLOCK_SIZE(")
    if macro_start < 0:
        raise RuntimeError("DISPATCH_BLOCK_SIZE macro not found")

    next_macro = text.find("#define ", macro_start + len("#define "))
    macro_end = next_macro if next_macro >= 0 else len(text)
    macro = text[macro_start:macro_end]

    if "block_size == 32" not in macro:
        err_pos = macro.find("Unsupported block_size")
        if err_pos < 0:
            raise RuntimeError("DISPATCH_BLOCK_SIZE unsupported fallback not found")

        prefix = macro[:err_pos]
        candidates = list(re.finditer(
            r'(?m)^(?P<indent>[ \t]*)\}[ \t]+else[ \t]*\{[ \t]*\\[ \t]*$',
            prefix,
        ))
        if not candidates:
            raise RuntimeError(
                "Could not locate DISPATCH_BLOCK_SIZE fallback else line"
            )

        fb = candidates[-1]
        indent = fb.group("indent")
        branch = (
            f"{indent}}} else if (block_size == 32) {{                                \\\n"
            f"{indent}  constexpr int BLOCK_SIZE = 32;                              \\\n"
            f"{indent}  __VA_ARGS__                                                 \\\n"
        )
        macro = macro[:fb.start()] + branch + macro[fb.start():]
        text = text[:macro_start] + macro + text[macro_end:]
        path.write_text(text, encoding="utf-8")

    verify = path.read_text(encoding="utf-8")
    vstart = verify.find("#define DISPATCH_BLOCK_SIZE(")
    vnext = verify.find("#define ", vstart + len("#define "))
    vmacro = verify[vstart:vnext if vnext >= 0 else len(verify)]

    required = {
        "branch32": "block_size == 32",
        "constexpr32": "constexpr int BLOCK_SIZE = 32;",
        "branch64": "block_size == 64",
        "constexpr64": "constexpr int BLOCK_SIZE = 64;",
        "branch128": "block_size == 128",
        "constexpr128": "constexpr int BLOCK_SIZE = 128;",
        "fallback": "Unsupported block_size",
    }
    missing = [name for name, token in required.items() if token not in vmacro]
    if missing:
        raise RuntimeError(f"BLKK32 macro verification failed: {missing}")
    if vmacro.count("block_size == 32") != 1:
        raise RuntimeError("Expected exactly one BLOCK_SIZE=32 dispatch branch")

    print("PATCH: dispatch_utils.h BLOCK_SIZE=32 macro branch verified")


def patch_fused_host_dispatch(src: Path):
    fused = src / "csrc" / "fused" / "fused.cu"
    pybind = src / "csrc" / "fused" / "pybind.cpp"
    if not fused.is_file() or not pybind.is_file():
        raise RuntimeError("Sage2.2 fused source files missing")

    marker = "SAGE_SM75_BLKK32_HOST_DISPATCH_V1_4"
    text = fused.read_text(encoding="utf-8")

    if marker not in text:
        include_anchor = "#include <cuda_bf16.h>\n"
        if include_anchor not in text:
            raise RuntimeError("fused.cu cuda_bf16 include anchor missing")

        local_macro = r'''
// SM75/Turing build marker + local dispatcher.
extern "C" __attribute__((used, visibility("default")))
const char SAGE_SM75_BLKK32_BUILD_MARKER[] =
    "SAGE_SM75_BLKK32_HOST_DISPATCH_V1_4";

#define DISPATCH_BLOCK_SIZE_SM75(block_size, BLOCK_SIZE, ...)    \
  if (block_size == 32) {                                        \
    constexpr int BLOCK_SIZE = 32;                               \
    __VA_ARGS__                                                  \
  } else if (block_size == 64) {                                 \
    constexpr int BLOCK_SIZE = 64;                               \
    __VA_ARGS__                                                  \
  } else if (block_size == 128) {                                \
    constexpr int BLOCK_SIZE = 128;                              \
    __VA_ARGS__                                                  \
  } else {                                                       \
    std::ostringstream err_msg;                                  \
    err_msg << "Unsupported block_size " << int(block_size);     \
    throw std::invalid_argument(err_msg.str());                  \
  }

'''
        text = text.replace(include_anchor, include_anchor + local_macro, 1)

    count_before = text.count("DISPATCH_BLOCK_SIZE(")
    if count_before:
        text = text.replace("DISPATCH_BLOCK_SIZE(", "DISPATCH_BLOCK_SIZE_SM75(")

    fused.write_text(text, encoding="utf-8")

    verify = fused.read_text(encoding="utf-8")
    if marker not in verify:
        raise RuntimeError("fused.cu build marker missing after patch")
    if "DISPATCH_BLOCK_SIZE(" in verify:
        raise RuntimeError("Unredirected DISPATCH_BLOCK_SIZE call remains in fused.cu")
    redirected = verify.count("DISPATCH_BLOCK_SIZE_SM75(")
    if redirected < 4:
        raise RuntimeError(
            f"Too few local block-size dispatcher occurrences in fused.cu: {redirected}"
        )
    for token in (
        "block_size == 32",
        "constexpr int BLOCK_SIZE = 32;",
        "block_size == 64",
        "block_size == 128",
    ):
        if token not in verify:
            raise RuntimeError(f"fused.cu verification missing: {token}")

    ptext = pybind.read_text(encoding="utf-8")
    if marker not in ptext:
        if "#include <string>" not in ptext:
            ptext = ptext.replace(
                "#include <torch/extension.h>\n",
                "#include <torch/extension.h>\n#include <string>\n",
                1,
            )
        anchor = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n{\n"
        if anchor not in ptext:
            raise RuntimeError("pybind module anchor missing")
        probe = (
            '  m.def("_sm75_blkk32_build_info", []() { '
            'return std::string("SAGE_SM75_BLKK32_HOST_DISPATCH_V1_4"); });\n'
        )
        ptext = ptext.replace(anchor, anchor + probe, 1)
        pybind.write_text(ptext, encoding="utf-8")

    pverify = pybind.read_text(encoding="utf-8")
    if "_sm75_blkk32_build_info" not in pverify or marker not in pverify:
        raise RuntimeError("pybind BLKK32 build probe patch failed")

    print(
        "PATCH: fused.cu explicit 32/64/128 host dispatcher; "
        f"redirected_call_sites={redirected - 1}"
    )


def binary_require_blkk32(fused_so: Path, env):
    marker = "SAGE_SM75_BLKK32_HOST_DISPATCH_V1_4"

    cp = run(["strings", fused_so], env=env, check=False)
    if cp.returncode != 0 or marker not in cp.stdout:
        raise RuntimeError(
            "Compiled _fused.so does not contain the v1.4 BLKK32 build marker"
        )

    if not re.search(r"QuantInt8Kernel.*ELj32E", cp.stdout):
        raise RuntimeError(
            "Compiled _fused.so has no visible QuantInt8Kernel BLOCK_SIZE=32 "
            "template instantiation"
        )

    cuobjdump_require_sm75(fused_so, env)

    code = (
        "import sageattention._fused as f;"
        "x=f._sm75_blkk32_build_info();"
        "print(x);"
        f"assert x == {marker!r}"
    )
    probe_env = env.copy()
    probe_env["PYTHONPATH"] = str(fused_so.parent.parent)
    run([sys.executable, "-c", code], env=probe_env)

    print("PREFLIGHT: compiled _fused BLKK32 host-dispatch probe PASS")

def patch_core(core: Path):
    text = core.read_text(encoding="utf-8")

    if 'if arch == "sm75":' not in text:
        anchor = '    if arch == "sm80":\n'
        if anchor not in text:
            raise RuntimeError("core.py public sm80 dispatch anchor not found")
        route = (
            '    if arch == "sm75":\n'
            '        return sageattn_qk_int8_pv_fp16_cuda('
            'q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, '
            'sm_scale=sm_scale, return_lse=return_lse, '
            'qk_quant_gran="per_warp", pv_accum_dtype="fp32")\n'
            '    elif arch == "sm80":\n'
        )
        text = text.replace(anchor, route, 1)

    original = (
        '        q_int8, q_scale, k_int8, k_scale = '
        'per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, '
        'BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and '
        'pv_accum_dtype == "fp16+fp32") else 32), BLKK=64)'
    )
    if 'BLKQ=64, WARPQ=16, BLKK=32' not in text:
        if original not in text:
            raise RuntimeError("core.py official per_warp geometry anchor not found")
        replacement = (
            '        if get_cuda_arch_versions()[q.device.index] == "sm75":\n'
            '            q_int8, q_scale, k_int8, k_scale = '
            'per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, '
            'BLKQ=64, WARPQ=16, BLKK=32)\n'
            '        else:\n'
            '            q_int8, q_scale, k_int8, k_scale = '
            'per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, '
            'BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and '
            'pv_accum_dtype == "fp16+fp32") else 32), BLKK=64)'
        )
        text = text.replace(original, replacement, 1)

    core.write_text(text, encoding="utf-8")
    syntax_check(core)

    verify = core.read_text(encoding="utf-8")
    for item in (
        'if arch == "sm75":',
        'qk_quant_gran="per_warp"',
        'pv_accum_dtype="fp32"',
        'BLKQ=64, WARPQ=16, BLKK=32',
    ):
        if item not in verify:
            raise RuntimeError(f"core.py verification missing: {item}")
    print("PATCH: core.py SM75 dispatch + BLKQ64/WARPQ16/BLKK32")


def unpack_wheel(wheel: Path, dest: Path):
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(wheel, "r") as zf:
        zf.extractall(dest)


def find_qattn_so(root: Path) -> Path:
    matches = sorted(root.rglob("*_qattn_sm80*.so"))
    if not matches:
        matches = sorted(root.rglob("*qattn_sm80*.so"))
    if not matches:
        raise RuntimeError("Turing qattn .so not found")
    return matches[0]


def regenerate_record(tree: Path):
    dist_infos = list(tree.glob(f"sageattention-{TARGET_VERSION}.dist-info"))
    if len(dist_infos) != 1:
        raise RuntimeError(f"Expected one target dist-info directory, found {dist_infos}")
    record = dist_infos[0] / "RECORD"

    rows = []
    for p in sorted(tree.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(tree).as_posix()
        if p == record:
            rows.append([rel, "", ""])
            continue
        data = p.read_bytes()
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()
        ).rstrip(b"=").decode()
        rows.append([rel, f"sha256={digest}", str(len(data))])

    with record.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def repack_wheel(tree: Path, output: Path):
    regenerate_record(tree)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(tree.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(tree).as_posix())


def cuobjdump_require_sm75(so: Path, env):
    tool = Path(env["CUDA_HOME"]) / "bin" / "cuobjdump"
    cp = run([tool, "--list-elf", so], env=env, check=False)
    if cp.returncode != 0 or not re.search(r"sm_?75", cp.stdout, re.I):
        raise RuntimeError(f"Native binary does not contain an SM75 cubin: {so}")
    print("PREFLIGHT: SM75 cubin confirmed:", so)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--sage22-ref", required=True)
    ap.add_argument("--turing-ref", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    ws = args.workspace.resolve()
    out = args.output.resolve()
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "CUDA_HOME": "/usr/local/cuda",
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
        "TORCH_CUDA_ARCH_LIST": "7.5",
        "MAX_JOBS": env.get("MAX_JOBS", "2"),
    })

    run([
        sys.executable, "-c",
        "import sys,torch;"
        "print(sys.version);print(torch.__version__);print(torch.version.cuda);"
        "assert sys.version_info[:2]==(3,12);"
        "assert torch.__version__.startswith('2.6.0');"
        "assert torch.version.cuda=='12.6'"
    ], env=env)
    run([Path(env["CUDA_HOME"]) / "bin" / "nvcc", "--version"], env=env)

    # Stage 1: proven Turing qattn under the exact ABI.
    turing = ws / "SageAttention2_Turing"
    run([
        "git", "clone", "--depth", "1", "--branch", args.turing_ref,
        "https://github.com/Ph0rk0z/SageAttention2.git", turing
    ])
    patch_turing_setup(turing / "setup.py")

    turing_dist = ws / "turing_dist"
    turing_dist.mkdir()
    run([
        sys.executable, "-m", "pip", "wheel", ".",
        "--no-build-isolation", "--no-deps", "-w", turing_dist
    ], cwd=turing, env=env)

    tw = list(turing_dist.glob("*.whl"))
    if len(tw) != 1:
        raise RuntimeError(f"Expected one Turing wheel, found {tw}")
    turing_tree = ws / "turing_tree"
    unpack_wheel(tw[0], turing_tree)
    qattn_so = find_qattn_so(turing_tree)
    cuobjdump_require_sm75(qattn_so, env)

    # Stage 2: Sage2.2 fused preprocessing for SM75.
    sage = ws / "SageAttention"
    run([
        "git", "clone", "--depth", "1", "--branch", args.sage22_ref,
        "https://github.com/thu-ml/SageAttention.git", sage
    ])
    patch_sage22_setup(sage / "setup.py")
    patch_blkk32(sage)
    patch_fused_host_dispatch(sage)
    patch_core(sage / "sageattention" / "core.py")

    run(
        ["git", "diff", "--", "setup.py", "sageattention/core.py", "csrc/dispatch_utils.h", "csrc/fused/fused.cu", "csrc/fused/pybind.cpp"],
        cwd=sage, env=env
    )

    for stale in (
        sage / "build",
        sage / "dist",
        sage / "sageattention.egg-info",
    ):
        if stale.exists():
            shutil.rmtree(stale)

    sage_dist = ws / "sage22_dist"
    sage_dist.mkdir()
    run([
        sys.executable, "-m", "pip", "wheel", ".",
        "--no-build-isolation", "--no-deps", "-w", sage_dist
    ], cwd=sage, env=env)

    sw = list(sage_dist.glob("*.whl"))
    if len(sw) != 1:
        raise RuntimeError(f"Expected one Sage2.2 wheel, found {sw}")
    if sw[0].name != TARGET_WHEEL:
        raise RuntimeError(
            f"Unexpected Sage2.2 wheel filename: {sw[0].name}; expected {TARGET_WHEEL}"
        )

    sage22_tree = ws / "sage22_verify_tree"
    unpack_wheel(sw[0], sage22_tree)
    sage22_fused = sorted((sage22_tree / "sageattention").glob("*_fused*.so"))
    if len(sage22_fused) != 1:
        raise RuntimeError(
            f"Expected exactly one Sage2.2 _fused extension, found {sage22_fused}"
        )
    binary_require_blkk32(sage22_fused[0], env)

    # Stage 3: integrate qattn into the Sage2.2-derived wheel.
    final_tree = ws / "final_tree"
    unpack_wheel(sw[0], final_tree)
    pkg = final_tree / "sageattention"
    if not pkg.is_dir():
        raise RuntimeError("sageattention package missing from Sage2.2 wheel")

    existing = list(pkg.glob("*_qattn_sm80*.so"))
    if existing:
        raise RuntimeError(
            f"Stock Sage2.2 qattn unexpectedly present for SM75 build: {existing}"
        )

    injected = pkg / qattn_so.name
    shutil.copy2(qattn_so, injected)
    cuobjdump_require_sm75(injected, env)

    dist_info = final_tree / f"sageattention-{TARGET_VERSION}.dist-info"
    if not dist_info.is_dir():
        raise RuntimeError("Target Sage2.2 dist-info directory missing")
    (dist_info / "SM75_BUILD.txt").write_text(
        "\n".join([
            "Integrated SageAttention 2.2-derived Turing/SM75 build",
            "Python: 3.12",
            "Torch: 2.6.0+cu126",
            "CUDA toolkit: 12.6",
            "GPU architecture: SM75",
            "Sage2.2 native INT8 preprocessing with explicit fused.cu BLOCK_SIZE=32 host dispatch",
            "Turing qattn: Ph0rk0z/SageAttention2",
            "Exact geometry: BLKQ=64, WARPQ=16, BLKK=32",
            "qattn ABI built under the same Python/Torch/CUDA stack",
            "BLKK32 binary probe: SAGE_SM75_BLKK32_HOST_DISPATCH_V1_4",
            "",
        ]),
        encoding="utf-8",
    )

    final = out / TARGET_WHEEL
    repack_wheel(final_tree, final)

    verify_tree = ws / "verify_tree"
    unpack_wheel(final, verify_tree)
    final_qattn = find_qattn_so(verify_tree)
    final_fused = sorted(verify_tree.rglob("*_fused*.so"))
    if not final_fused:
        raise RuntimeError("Final wheel is missing Sage2.2 _fused native extension")
    cuobjdump_require_sm75(final_qattn, env)
    binary_require_blkk32(final_fused[0], env)
    syntax_check(verify_tree / "sageattention" / "core.py")

    print("\nFINAL:", final)
    print("SIZE:", final.stat().st_size)
    print("SHA256:", hashlib.sha256(final.read_bytes()).hexdigest())
    print("BUILD ARCHITECTURE: Sage2.2 fused preprocessing + proven Turing qattn")
    print("NOTE: Real T4 runtime validation is still required after GitHub build.")


if __name__ == "__main__":
    main()
