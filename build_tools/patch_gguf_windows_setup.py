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
