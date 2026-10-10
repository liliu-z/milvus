#!/usr/bin/env bash
# Run after the native libraries and headers have been installed by core_build.sh.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
deps=${COLD_SEARCH_DEPS:-"$repo/.cold-search-deps"}
python3 tools/cold_search/prepare.py go --output "$deps"
source scripts/setenv.sh
set -e
export GOWORK=off
export GOFLAGS="-modfile=$deps/milvus.mod"
export CC=${CC:-gcc-12} CXX=${CXX:-g++-12}
mkdir -p "$deps/tmp" "$deps/bin"
export TMPDIR=${TMPDIR:-"$deps/tmp"} GOTMPDIR=${GOTMPDIR:-"$deps/tmp"}
"${GO:-go}" build -p "${BUILD_JOBS:-5}" -pgo=auto \
  -tags dynamic,sonic,with_jemalloc,bytedance_tango \
  -ldflags="-checklinkname=0 -X github.com/milvus-io/milvus/cmd/milvus.GitCommit=$(git rev-parse HEAD)" \
  -o "$deps/bin/milvus" ./cmd/main.go
