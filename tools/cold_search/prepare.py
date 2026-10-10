#!/usr/bin/env python3
"""Prepare pinned, isolated dependency overlays without editing module/Conan caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PATCHES = Path(__file__).resolve().parent / 'dependencies'
MANIFEST = json.loads((PATCHES / 'manifest.json').read_text())


def run(args, cwd=ROOT, **kwargs):
    return subprocess.run([str(x) for x in args], cwd=cwd, check=True, **kwargs)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(name, destination, populate):
    spec = MANIFEST[name]
    stamp = digest(json.dumps(spec, sort_keys=True).encode() + b''.join(
        (PATCHES / p).read_bytes() for p in spec['patches']))
    marker = destination / '.cold-search-patchset'
    if destination.exists():
        if marker.exists() and marker.read_text().strip() == stamp:
            return
        raise RuntimeError(f'{destination} already exists with a different patchset; use a new --output directory')
    # Publish only a fully patched directory. Never replace someone else's source.
    with tempfile.TemporaryDirectory(prefix=name + '-', dir=destination.parent) as tmp:
        source = Path(tmp) / 'source'
        populate(source)
        for patch in spec['patches']:
            run(['patch', '--batch', '--forward', '-p1', '-i', PATCHES / patch], cwd=source)
        (source / '.cold-search-patchset').write_text(stamp + '\n')
        source.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component', choices=['go', 'storage'])
    parser.add_argument('--output', type=Path, default=ROOT / '.cold-search-deps')
    parser.add_argument('--go', default=os.environ.get('GO', 'go'))
    parser.add_argument('--storage-source', type=Path, help='optional existing Git checkout; archive the pinned commit, ignoring local changes')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.component == 'storage':
        spec = MANIFEST['milvus-storage']
        def populate(dest):
            if args.storage_source:
                with tempfile.TemporaryFile() as archive:
                    run(['git', 'archive', spec['revision']], cwd=args.storage_source.resolve(), stdout=archive)
                    archive.seek(0)
                    dest.mkdir()
                    with tarfile.open(fileobj=archive) as tar:
                        tar.extractall(dest, filter='data')
            else:
                run(['git', 'clone', '--filter=blob:none', '--no-checkout', spec['repository'], dest])
                run(['git', 'checkout', '--detach', spec['revision']], cwd=dest)
        prepare('milvus-storage', output / 'milvus-storage', populate)
        print(f'-DFETCHCONTENT_SOURCE_DIR_MILVUS-STORAGE={output / "milvus-storage"}')
        return
    env = {**os.environ, 'GOWORK': 'off'}
    env.pop('GOFLAGS', None)
    replacements = {}
    for name in ('grpc', 'woodpecker'):
        spec = MANIFEST[name]
        if not spec['patches']:
            continue
        downloaded = json.loads(subprocess.check_output(
            [args.go, 'mod', 'download', '-json', spec['module'] + '@' + spec['version']], cwd=ROOT, env=env))
        def populate(dest, original=Path(downloaded['Dir'])):
            shutil.copytree(original, dest)
            for path in [dest, *dest.rglob('*')]:
                if not path.is_symlink():
                    path.chmod(path.stat().st_mode | 0o200)
        prepare(name, output / name, populate)
        replacements[spec['module']] = output / name
    modfile = output / 'milvus.mod'
    shutil.copyfile(ROOT / 'go.mod', modfile)
    shutil.copyfile(ROOT / 'go.sum', modfile.with_suffix('.sum'))
    replacements.update({
        'github.com/milvus-io/milvus/pkg/v3': ROOT / 'pkg',
        'github.com/milvus-io/milvus/client/v3': ROOT / 'client',
    })
    run([args.go, 'mod', 'edit', '-modfile=' + str(modfile),
         *[f'-replace={module}={path}' for module, path in replacements.items()]], env=env)
    print(f'GOWORK=off GOFLAGS=-modfile={modfile}')


if __name__ == '__main__':
    main()
