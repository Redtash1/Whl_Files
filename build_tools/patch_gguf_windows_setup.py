from __future__ import annotations

from pathlib import Path
import re
import sys

MSVC_OLD = r'''def _vc_tools_dir() -> str:
    base = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC")
    entries = [entry for entry in base.iterdir() if entry.is_dir()]
    if not entries:
        raise FileNotFoundError(f"Missing MSVC tools in {base}")
    return str(sorted(entries)[-1])
'''

MSVC_NEW = r'''def _vc_tools_dir() -> str:
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

    raise FileNotFoundError(
        "Could not locate Visual Studio 2022 MSVC tools under " + str(vs_root)
    )
'''

def patch_msvc(text: str) -> str:
    # Fresh upstream source: patch Community-only implementation.
    if MSVC_OLD in text:
        return text.replace(MSVC_OLD, MSVC_NEW, 1)

    # Idempotent rerun against an already-patched tree.
    if (
        'os.environ.get("VCToolsInstallDir"' in text
        and '"Enterprise", "Community", "Professional", "BuildTools"' in text
    ):
        return text

    raise RuntimeError(
        "Exact upstream _vc_tools_dir() anchor not found and patched markers are absent; "
        "refusing blind MSVC edit."
    )

def patch_python_requires(text: str) -> str:
    # Limit this edit strictly to python_requires metadata.
    # Accept either quote style and either ordering of the two specifiers.
    patterns = (
        (
            r'(python_requires\s*=\s*["\'])>=3\.10,<3\.12(["\'])',
            r'\1>=3.10,<3.13\2',
        ),
        (
            r'(python_requires\s*=\s*["\'])<3\.12,>=3\.10(["\'])',
            r'\1<3.13,>=3.10\2',
        ),
    )

    for pattern, replacement in patterns:
        new_text, count = re.subn(pattern, replacement, text, count=1)
        if count == 1:
            return new_text

    # Idempotent rerun.
    if re.search(
        r'python_requires\s*=\s*["\'](?:>=3\.10,<3\.13|<3\.13,>=3\.10)["\']',
        text,
    ):
        return text

    raise RuntimeError(
        "Expected upstream python_requires gate (>=3.10,<3.12 or <3.12,>=3.10) "
        "was not found; refusing blind metadata edit."
    )

def verify(text: str) -> None:
    compile(text, "patched_setup.py", "exec")

    required = (
        'os.environ.get("VCToolsInstallDir"',
        '"Enterprise", "Community", "Professional", "BuildTools"',
    )
    for marker in required:
        if marker not in text:
            raise RuntimeError(f"Patched setup.py missing marker: {marker}")

    if not re.search(
        r'python_requires\s*=\s*["\'](?:>=3\.10,<3\.13|<3\.13,>=3\.10)["\']',
        text,
    ):
        raise RuntimeError("Patched setup.py does not permit Python 3.12")

    if re.search(
        r'python_requires\s*=\s*["\'](?:>=3\.10,<3\.12|<3\.12,>=3\.10)["\']',
        text,
    ):
        raise RuntimeError("Old Python <3.12 gate still present")

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_gguf_windows_setup.py <setup.py>")

    path = Path(sys.argv[1]).resolve()
    if not path.is_file():
        raise SystemExit(f"Missing setup.py: {path}")

    text = path.read_text(encoding="utf-8")
    text = patch_msvc(text)
    text = patch_python_requires(text)
    verify(text)
    path.write_text(text, encoding="utf-8")

    print("Patched upstream setup.py:")
    print("  MSVC discovery: VS2022 Enterprise/Community/Professional/BuildTools")
    print("  Python metadata: Python 3.12 permitted (python_requires <3.13)")

if __name__ == "__main__":
    main()
