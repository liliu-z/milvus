#!/usr/bin/env bash
# Foreground execution; the caller owns process supervision and its PID file.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
: "${MILVUSCONF:?run render-config.py and set MILVUSCONF}"
source scripts/setenv.sh
set -e
source tools/cold_search/runtime.env.example
binary=${1:-"$repo/.cold-search-deps/bin/milvus"}
exec taskset -c "$MILVUS_CPUS" "$binary" run standalone
