from pathlib import Path
import sys
OLD = '''def _vc_tools_dir() -> str:\n    base = Path(r"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC")\n    entries = [entry for entry in base.iterdir() if entry.is_dir()]\n    if not entries:\n        raise FileNotFoundError(f"Missing MSVC tools in {base}")\n    return str(sorted(entries)[-1])\n'''
NEW = '''def _vc_tools_dir() -> str:\n    configured = os.environ.get("VCToolsInstallDir", "").strip()\n    if configured:\n        candidate = Path(configured)\n        if candidate.is_dir():\n            return str(candidate)\n    vs_root = Path(r"C:\\Program Files\\Microsoft Visual Studio\\2022")\n    for edition in ("Enterprise", "Community", "Professional", "BuildTools"):\n        base = vs_root / edition / "VC" / "Tools" / "MSVC"\n        if not base.is_dir():\n            continue\n        entries = [entry for entry in base.iterdir() if entry.is_dir()]\n        if entries:\n            return str(sorted(entries)[-1])\n    raise FileNotFoundError("Could not locate Visual Studio 2022 MSVC tools under " + str(vs_root))\n'''
def main():
    if len(sys.argv) != 2: raise SystemExit("usage: patch_gguf_windows_setup.py <setup.py>")
    p=Path(sys.argv[1]); text=p.read_text(encoding="utf-8")
    if OLD in text: text=text.replace(OLD,NEW,1)
    elif 'os.environ.get("VCToolsInstallDir"' not in text: raise RuntimeError("Exact upstream MSVC anchor not found; refusing blind patch")
    compile(text,str(p),"exec"); p.write_text(text,encoding="utf-8")
    print("PASS: setup.py MSVC discovery patched")
if __name__ == "__main__": main()

# v1.6: explicitly forward VS/Windows SDK INCLUDE and LIB paths into
# Torch CUDAExtension. v1.5 proved corecrt.h exists, but nvcc's host CL
# invocation did not inherit the UCRT search path.
text = setup_path.read_text(encoding="utf-8")

old_env_lists = (
    'EXTRA_INCLUDE_DIRS = _env_path_list("LLAMACPP_GGUF_CUDA_INCLUDE_DIRS")\n'
    'EXTRA_LIBRARY_DIRS = _env_path_list("LLAMACPP_GGUF_CUDA_LIB_DIRS")\n'
)
new_env_lists = (
    'EXTRA_INCLUDE_DIRS = _env_path_list("LLAMACPP_GGUF_CUDA_INCLUDE_DIRS")\n'
    'EXTRA_LIBRARY_DIRS = _env_path_list("LLAMACPP_GGUF_CUDA_LIB_DIRS")\n'
    'if os.name == "nt":\n'
    '    EXTRA_INCLUDE_DIRS += _env_path_list("INCLUDE")\n'
    '    EXTRA_LIBRARY_DIRS += _env_path_list("LIB")\n'
)
if text.count(old_env_lists) != 1:
    raise RuntimeError("Expected EXTRA include/library anchor exactly once")
text = text.replace(old_env_lists, new_env_lists, 1)

old_attention = (
    '    CUDAExtension(\n'
    '        name="llamacpp_gguf_cuda._attention",\n'
    '        sources=[str(CSRC / "q8_paged_attention_bindings.cpp"), str(CSRC / "sm120_bindings.cpp"), str(CSRC / "q8_paged_attention.cu")],\n'
    '        extra_compile_args=extra_compile_args,\n'
    '        extra_link_args=EXTRA_LINK_ARGS,\n'
    '        library_dirs=EXTRA_LIBRARY_DIRS,\n'
    '        libraries=["cuda"],\n'
    '    ),\n'
)
new_attention = (
    '    CUDAExtension(\n'
    '        name="llamacpp_gguf_cuda._attention",\n'
    '        sources=[str(CSRC / "q8_paged_attention_bindings.cpp"), str(CSRC / "sm120_bindings.cpp"), str(CSRC / "q8_paged_attention.cu")],\n'
    '        include_dirs=EXTRA_INCLUDE_DIRS,\n'
    '        extra_compile_args=extra_compile_args,\n'
    '        extra_link_args=EXTRA_LINK_ARGS,\n'
    '        library_dirs=EXTRA_LIBRARY_DIRS,\n'
    '        libraries=["cuda"],\n'
    '    ),\n'
)
if text.count(old_attention) != 1:
    raise RuntimeError("Expected _attention CUDAExtension anchor exactly once")
text = text.replace(old_attention, new_attention, 1)

setup_path.write_text(text, encoding="utf-8")
check = setup_path.read_text(encoding="utf-8")
if 'EXTRA_INCLUDE_DIRS += _env_path_list("INCLUDE")' not in check:
    raise RuntimeError("Windows INCLUDE forwarding patch missing")
if 'EXTRA_LIBRARY_DIRS += _env_path_list("LIB")' not in check:
    raise RuntimeError("Windows LIB forwarding patch missing")
if check.count("include_dirs=EXTRA_INCLUDE_DIRS") != 1:
    raise RuntimeError("_attention include_dirs patch missing or duplicated")
print("[patch] Windows SDK/UCRT paths forwarded explicitly to CUDAExtension")
