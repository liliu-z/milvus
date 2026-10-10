#!/usr/bin/env python3
"""Build focused native checks using the configured Milvus Ninja dependency graph."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('check', choices=['small-parquet', 'scoped-prefetch', 'immutable-vector', 'column-group', 'int64-pk'])
p.add_argument('--native-build', type=Path, required=True)
p.add_argument('--gtest-prefix', type=Path, required=True, help='Conan gtest package containing include/ and lib/libgtest.a')
p.add_argument('--output', type=Path, default=ROOT / '.cold-search-deps/checks')
a = p.parse_args()
build = a.native_build.resolve()
output = a.output.resolve()
output.mkdir(parents=True, exist_ok=True)
entries = json.loads((build / 'compile_commands.json').read_text())
entry = next(x for x in entries if x['file'].endswith('/segcore/ChunkedSegmentSealedImpl.cpp'))
args = entry.get('arguments') or shlex.split(entry['command'])
clean = []
i = 0
while i < len(args):
    if args[i] in ('-o', '-MF', '-MT', '-MQ', '-include'):
        i += 2
    elif args[i] in ('-c', '-MD', '-MMD') or args[i] == entry['file']:
        i += 1
    else:
        clean.append(args[i])
        i += 1
prefix = a.gtest_prefix.resolve()
clean += ['-O0', '-DMILVUS_UNIT_TEST', f'-DMILVUS_CPPUT_OUTPUT_DIR="{output}"',
          '-I' + str(prefix / 'include'), '-I' + str(ROOT / 'internal/core/unittest'),
          '-I' + str(ROOT / 'internal/core/output/include')]
tests = {
    'immutable-vector': 'src/common/ImmutableVectorChunkTest.cpp',
    'column-group': 'src/mmap/ChunkedColumnGroupTest.cpp',
    'int64-pk': 'src/segcore/Int64PrimaryKeyFillTest.cpp',
}
if a.check in tests:
    sources = [ROOT / 'internal/core' / tests[a.check], ROOT / 'internal/core/unittest/init_gtest.cpp']
else:
    sources = [HERE / 'checks' / (a.check + '-check.cpp')]
sources.append(HERE / 'checks/cpp-log-stub.cpp')
objects = []
for source in sources:
    obj = output / (source.stem + '.o')
    subprocess.run(clean + ['-c', str(source), '-o', str(obj)], cwd=build, check=True)
    objects.append(str(obj))
lines = (build / 'build.ninja').read_text().splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith('build src/libmilvus_core.so:'))
link = next(line.split(' = ', 1)[1] for line in lines[start + 1:] if line.startswith('  LINK_LIBRARIES = '))
binary = output / a.check
subprocess.run([clean[0], '-fuse-ld=lld', '-fopenmp', '-o', str(binary), *objects,
                str(build / 'src/libmilvus_core.so'), str(prefix / 'lib/libgtest.a'), *shlex.split(link)], cwd=build, check=True)
print(binary)
print('Set LD_LIBRARY_PATH to the matching installed native libraries before running.')
if a.check not in tests:
    print('Pass an absolute path to a new disposable Parquet file as the only argument.')
