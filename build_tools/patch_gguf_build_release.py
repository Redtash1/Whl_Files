from __future__ import annotations

from pathlib import Path
import re
import sys

TARGET_SUFFIX = "+torch2.12cu130"

def patch_build_release(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    text, n = re.subn(
        r"parser\.add_argument\('--target', choices=\('py310', 'py311'\), required=True\)",
        "parser.add_argument('--target', choices=('py310', 'py311', 'py312'), required=True)",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not patch build_release.py target choices")

    old_map = (
        "expected = {'py310': ((3, 10), '2.7.1', '12.8', '+torch271cu128py310'), "
        "'py311': ((3, 11), '2.10.0', '13.0', '+torch210cu130py311')}[args.target]"
    )
    new_map = (
        "expected = {'py310': ((3, 10), '2.7.1', '12.8', '+torch271cu128py310'), "
        "'py311': ((3, 11), '2.10.0', '13.0', '+torch210cu130py311'), "
        "'py312': ((3, 12), '2.12.0', '13.0', '+torch2.12cu130')}[args.target]"
    )
    if old_map not in text:
        raise RuntimeError("Could not find exact build_release.py expected-target map")
    text = text.replace(old_map, new_map, 1)

    compile(text, str(path), "exec")
    for needle in (
        "'py312'",
        "((3, 12), '2.12.0', '13.0', '+torch2.12cu130')",
        "environment.pop('TORCH_CUDA_ARCH_LIST', None)",
        "'pip', 'wheel'",
        "'--no-build-isolation'",
        "'--no-deps'",
    ):
        if needle not in text:
            raise RuntimeError(f"Patched helper missing marker: {needle}")

    path.write_text(text, encoding="utf-8")
    print("Patched upstream release helper for Python 3.12 / Torch 2.12.0 / cu130")
    print("Version suffix:", TARGET_SUFFIX)

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_gguf_build_release.py <build_release.py>")
    p = Path(sys.argv[1]).resolve()
    if not p.is_file():
        raise SystemExit(f"Missing helper: {p}")
    patch_build_release(p)

if __name__ == "__main__":
    main()
