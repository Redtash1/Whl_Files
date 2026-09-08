from pathlib import Path
import sys
OLD='requires-python = ">=3.10,<3.12"'
NEW='requires-python = ">=3.10,<3.13"'
def main():
    if len(sys.argv) != 2: raise SystemExit("usage: patch_gguf_pyproject_py312.py <pyproject.toml>")
    p=Path(sys.argv[1]); text=p.read_text(encoding="utf-8"); n=text.count(OLD)
    if n == 1: text=text.replace(OLD,NEW,1)
    elif n == 0 and NEW in text: pass
    else: raise RuntimeError(f"Expected exactly one upstream Python gate; found {n}")
    if OLD in text or NEW not in text: raise RuntimeError("Python metadata patch verification failed")
    p.write_text(text,encoding="utf-8"); print("PASS: pyproject.toml Python 3.12 gate patched")
if __name__ == "__main__": main()
