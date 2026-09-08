from pathlib import Path
import sys

OLD_VC = r"""def _vc_tools_dir() -> str:
    base = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC")
    entries = [entry for entry in base.iterdir() if entry.is_dir()]
    if not entries:
        raise FileNotFoundError(f"Missing MSVC tools in {base}")
    return str(sorted(entries)[-1])
"""
NEW_VC = r"""def _vc_tools_dir() -> str:
    configured = os.environ.get("VCToolsInstallDir", "").strip()
    if configured:
        candidate = Path(configured)
        if candidate.is_dir():
            return str(candidate)
    vs_root = Path(r"C:\Program Files\Microsoft Visual Studio\2022")
    for edition in ("Enterprise", "Community", "Professional", "BuildTools"):
        base = vs_root / edition / "VC" / "Tools" / "MSVC"
        if not base.is_dir():
            continue
        entries = [entry for entry in base.iterdir() if entry.is_dir()]
        if entries:
            return str(sorted(entries)[-1])
    raise FileNotFoundError("Could not locate Visual Studio 2022 MSVC tools under " + str(vs_root))
"""
OLD_ENV = (
 'EXTRA_INCLUDE_DIRS = _env_path_list("LLAMACPP_GGUF_CUDA_INCLUDE_DIRS")\n'
 'EXTRA_LIBRARY_DIRS = _env_path_list("LLAMACPP_GGUF_CUDA_LIB_DIRS")\n'
)
NEW_ENV = OLD_ENV + (
 'if os.name == "nt":\n'
 '    EXTRA_INCLUDE_DIRS += _env_path_list("INCLUDE")\n'
 '    EXTRA_LIBRARY_DIRS += _env_path_list("LIB")\n'
)
OLD_ATTN = (
 '    CUDAExtension(\n'
 '        name="llamacpp_gguf_cuda._attention",\n'
 '        sources=[str(CSRC / "q8_paged_attention_bindings.cpp"), str(CSRC / "sm120_bindings.cpp"), str(CSRC / "q8_paged_attention.cu")],\n'
 '        extra_compile_args=extra_compile_args,\n'
 '        extra_link_args=EXTRA_LINK_ARGS,\n'
 '        library_dirs=EXTRA_LIBRARY_DIRS,\n'
 '        libraries=["cuda"],\n'
 '    ),\n'
)
NEW_ATTN = OLD_ATTN.replace(
 '        extra_compile_args=extra_compile_args,\n',
 '        include_dirs=EXTRA_INCLUDE_DIRS,\n        extra_compile_args=extra_compile_args,\n',1)

def once(text, old, new, label):
    if text.count(old)==1:
        return text.replace(old,new,1)
    if old not in text and new in text:
        return text
    raise RuntimeError(f"{label}: exact source anchor missing/duplicated")

def main():
    if len(sys.argv)!=2:
        raise SystemExit("usage: patch_gguf_windows_setup.py <setup.py>")
    p=Path(sys.argv[1])
    text=p.read_text(encoding="utf-8")
    text=once(text,OLD_VC,NEW_VC,"MSVC")
    text=once(text,OLD_ENV,NEW_ENV,"INCLUDE/LIB")
    text=once(text,OLD_ATTN,NEW_ATTN,"_attention")
    compile(text,str(p),"exec")
    p.write_text(text,encoding="utf-8")
    check=p.read_text(encoding="utf-8")
    for marker in (
        'os.environ.get("VCToolsInstallDir"',
        'EXTRA_INCLUDE_DIRS += _env_path_list("INCLUDE")',
        'EXTRA_LIBRARY_DIRS += _env_path_list("LIB")',
        'include_dirs=EXTRA_INCLUDE_DIRS',
    ):
        if marker not in check:
            raise RuntimeError("post-patch marker missing: "+marker)
    print("PASS: complete Windows setup.py patch applied")
if __name__=="__main__":
    main()
