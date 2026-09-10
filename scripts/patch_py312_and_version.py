from __future__ import annotations
import pathlib, re, sys

root = pathlib.Path(sys.argv[1]).resolve()
files = [p for p in [root/'pyproject.toml', root/'setup.py', root/'setup.cfg'] if p.exists()]
if not files:
    raise SystemExit('No package metadata file found')

changed = []
for p in files:
    text = p.read_text(encoding='utf-8')
    orig = text
    # Upstream metadata historically restricted Python to <3.12. Permit 3.12.
    text = text.replace('>=3.10,<3.12', '>=3.10,<3.13')
    text = text.replace('>=3.10, <3.12', '>=3.10, <3.13')
    text = text.replace('<3.12', '<3.13')
    # Give this private compatibility wheel the explicit project-local version.
    text = re.sub(r'(?m)^(version\s*=\s*["\'])1\.0\.21[^"\']*(["\'])',
                  r'\g<1>1.0.21+torch2.12cu130\2', text)
    if text != orig:
        p.write_text(text, encoding='utf-8')
        changed.append(str(p))

print('Patched metadata files:', changed)
for p in files:
    print(f'--- {p.name} ---')
    for line in p.read_text(encoding='utf-8').splitlines():
        if 'python' in line.lower() or 'version' in line.lower():
            print(line)
