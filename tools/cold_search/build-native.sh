#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
deps=${COLD_SEARCH_DEPS:-"$repo/.cold-search-deps"}
python3 tools/cold_search/prepare.py storage --output "$deps" "$@"
export CC=${CC:-gcc-12} CXX=${CXX:-g++-12}
export jobs=${BUILD_JOBS:-5} CARGO_BUILD_JOBS=${BUILD_JOBS:-5}
export MILVUS_CARGO_TARGET_ROOT=${MILVUS_CARGO_TARGET_ROOT:-"$deps/cargo"}
export CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS:-} -DFETCHCONTENT_SOURCE_DIR_MILVUS-STORAGE=$deps/milvus-storage"
# Select a profile appropriate to the host. The supplied example is Linux ARM64/GCC12.
profile=${COLD_SEARCH_CONAN_PROFILE:-"$repo/tools/cold_search/conan-arm64-gcc12.profile"}
conan install internal/core --output-folder cmake_build/conan \
  --profile:host="$profile" --profile:build="$profile" --build=missing
bash scripts/core_build.sh -t Release
# AWS is an isolated second link step: use build-aws-overlay.py with this Conan
# build's source/build/package paths, then install BOTH resulting native DSOs.
