from pathlib import Path
import sys

OLD = '''def _vc_tools_dir() -> str:
    base = Path(r"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC")
    entries = [entry for entry in base.iterdir() if entry.is_dir()]
    if not entries:
        raise FileNotFoundError(f"Missing MSVC tools in {base}")
    return str(sorted(entries)[-1])
'''

NEW = '''def _vc_tools_dir() -> str:
    configured = os.environ.get("VCToolsInstallDir", "").strip()
    if configured:
        candidate = Path(configured)
        if candidate.is_dir():
            return str(candidate)

    vs_root = Path(r"C:\\Program Files\\Microsoft Visual Studio\\2022")
    for edition in ("Enterprise", "Community", "Professional", "BuildTools"):
        base = vs_root / edition / "VC" / "Tools" / "MSVC"
        if not base.is_dir():
            continue
        entries = [entry for entry in base.iterdir() if entry.is_dir()]
        if entries:
            return str(sorted(entries)[-1])

    raise FileNotFoundError("Could not locate Visual Studio 2022 MSVC tools under " + str(vs_root))
'''

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_gguf_windows_setup.py <setup.py>")
    path = Path(sys.argv[1]).resolve()
    text = path.read_text(encoding="utf-8")
    if OLD not in text:
        raise RuntimeError("Exact upstream _vc_tools_dir() anchor not found; refusing blind patch")
    text = text.replace(OLD, NEW, 1)
    compile(text, str(path), "exec")
    for marker in ('VCToolsInstallDir', '"Enterprise", "Community", "Professional", "BuildTools"'):
        if marker not in text:
            raise RuntimeError("Missing patched marker: " + marker)
    path.write_text(text, encoding="utf-8")
    print("Patched upstream setup.py MSVC discovery for VS2022 editions.")

if __name__ == "__main__":
    main()
