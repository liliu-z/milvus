# Cold-search experiment tools

Start with the [Chinese technical report](../../docs/performance/cold-search/REPORT.zh-CN.md) and [experiment work log](../../docs/performance/cold-search/WORKLOG.zh-CN.md). They define the two output scenarios, Release-cold protocol, measured limits, dependency/build sequence and evidence.

- `prepare.py`: pinned private Go/storage overlays; never patches shared module caches.
- `build-native.sh`, `build-aws-overlay.py`, `build-go.sh`: native, isolated AWS archive, and Go build entrypoints. Deploy both native DSOs together.
- `build-check.py`: focused native checks using an existing Ninja compile/link graph; requires GTest and `ld.lld` on PATH.
- `compose.yaml`, `profile.yaml`, `runtime.env.example`, `render-config.py`, `run.sh`: local MinIO/etcd and the measured standalone profile. Set a dedicated data root and adjust CPU/memory for your host.
- `bench.py`: deterministic 1000 × 768 dataset, five scalars, ID or ID + vector only. Setup flush is outside the timer. Default Release wait is 62 seconds.
- `collector.py`, `metric_delta.py`, `extract_all.py`, `summarize_scenarios.py`, `run-cohort.py`: fixed-window capture and analysis. Set the same `COLD_SEARCH_WORK` for all processes; the default is `.cold-search-results` under the repository.

`mc admin trace` capture is required for object-access counts. Trace interval-connected waves are not causal dispatch counts when speculative reads and foreground fallback overlap. Inspect repeated keys and consumed/reserved counters as described in the report.

Profiles and dependency APIs are experimental. Ordinary upstream gRPC does not provide `ServeInProcess`; use the generated modfile. Reusing an overlay prepared for a different patchset fails explicitly; choose a fresh `--output` directory. Measured latency samples do not constitute a production P99 guarantee.
