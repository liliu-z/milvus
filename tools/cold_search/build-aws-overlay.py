#!/usr/bin/env python3
"""Rebuild the patched AWS CurlHttpClient in a private static archive and Ninja file.

Use a configured Conan 2 AWS SDK build of the pinned recipe. This deliberately
leaves its sources, flags, objects and installed package unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--aws-source', type=Path, required=True, help='SDK root containing src/aws-cpp-sdk-core')
p.add_argument('--aws-build', type=Path, required=True, help='directory containing CMakeFiles/aws-cpp-sdk-core.dir/flags.make')
p.add_argument('--aws-archive', type=Path, required=True)
p.add_argument('--native-build', type=Path, required=True)
p.add_argument('--ninja-file', default='build.ninja', help='configured native Ninja input, relative to --native-build')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cxx', default=os.environ.get('CXX', 'g++-12'))
a = p.parse_args()
spec = json.loads((HERE / 'dependencies/manifest.json').read_text())['aws-sdk-cpp']
source = a.aws_source.resolve() / spec['source_file']
assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['source_sha256'], 'AWS input differs from the tested source'
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
patched_source = a.output / 'source' / spec['source_file']
patched_source.parent.mkdir(parents=True)
shutil.copyfile(source, patched_source)
for name in spec['patches']:
    subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(HERE / 'dependencies' / name)],
                   cwd=a.output / 'source', check=True)
flags = {}
for line in (a.aws_build / 'CMakeFiles/aws-cpp-sdk-core.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        key, value = line.split(' = ', 1)
        flags[key] = shlex.split(value)
obj = a.output / 'CurlHttpClient.cpp.o'
subprocess.run([a.cxx, *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
                '-c', str(patched_source), '-o', str(obj)], cwd=a.aws_build.resolve(), check=True)
archive = a.output / a.aws_archive.name
shutil.copyfile(a.aws_archive, archive)
subprocess.run(['ar', 'r', str(archive), str(obj)], check=True)
build = a.native_build.resolve()
original = (build / a.ninja_file).read_text()
original_archive = str(a.aws_archive.resolve())
assert original_archive in original, 'Ninja input must link the exact original AWS archive'
assert not any(c in str(archive) for c in (' ', '$', '\n')), 'Use an overlay path without Ninja-special characters'
target = build / 'build-cold-search-aws.ninja'
assert not target.exists(), 'Use a fresh Ninja overlay name/build directory'
target.write_text(original.replace(original_archive, str(archive)))
print(f'ninja -C {shlex.quote(str(build))} -f {target.name} -j5 src/libmilvus_core.so')
print('Rebuild and deploy BOTH libmilvus_core.so and libmilvus-storage.so before measuring.')
