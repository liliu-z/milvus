#!/usr/bin/env python3
"""Render a private config directory; never overwrite the repository defaults."""
import argparse
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = a.data_root.resolve()
# YAML plain scalars in the template need a simple absolute filesystem path.
if any(c in str(root) for c in '\n\r\t:#'):
    p.error('data-root contains characters that need YAML quoting')
a.output.mkdir(parents=True, exist_ok=False)
for source in (ROOT / 'configs').iterdir():
    if source.is_file():
        shutil.copyfile(source, a.output / source.name)
(a.output / 'user.yaml').write_text((HERE / 'profile.yaml').read_text().replace('${COLD_SEARCH_ROOT}', str(root)))
print(f'MILVUSCONF={a.output.resolve()}')
