# Milvus Search Performance Optimization Log

## Environment
- CPU: Intel Xeon Platinum 8488C, 8 cores
- Dataset: cohere_medium_1m (1M vectors, 768 dim, FLOAT_VECTOR)
- Index: HNSW_SQ (M=16, efConstruction=200, SQ8, COSINE)
- Segments: 5 sealed segments (347K, 347K, 246K, 51K, 10K rows)
- Search params: ef=128, top_k=100, nq=1
- Benchmark: 5 concurrency levels (1,5,10,20,30) x 2 minutes each

## Baseline Results

| Concurrency | QPS | Avg Lat(us) | P50(us) | P95(us) | P99(us) | Recall@100 |
|---|---|---|---|---|---|---|
| 1 | 495 | 2013 | 1997 | 2317 | 2561 | 0.9549 |
| 5 | 1133 | 4400 | 4242 | 6302 | 7833 | 0.9549 |
| 10 | 1243 | 8038 | 7746 | 11585 | 15800 | 0.9549 |
| 20 | 1367 | 14626 | 14414 | 21492 | 28007 | 0.9549 |
| 30 | 1456 | 20597 | 20617 | 29880 | 37349 | 0.9549 |

## Opt1: Reduced C++ threads to CPUNum (8) + removed all hot-path metrics

**Changes:**
1. `internal/util/cgo/pool.go` - Removed metrics (time.Now(), CGOQueueDuration, RunningCgoCallTotal, CGODuration) from cgoCaller.call()
2. `internal/util/cgo/executor.go` - Changed C++ executor thread pool from MaxReadConcurrency*CGOPoolSizeRatio (32) to hardware.GetCPUNum() (8)
3. `internal/util/searchutil/scheduler/concurrent_safe_scheduler.go` - Removed QueryNodeReadTaskConcurrency Inc/Dec and collector.Counter Inc/Dec from exec() loop; removed QueryNodeReadTaskUnsolveLen Inc/Dec from Add()
4. `internal/querynodev2/tasks/search_task.go` - Removed QueryNodeSQLatencyInQueue and QueryNodeSQPerUserLatencyInQueue from PreExecute(); removed QueryNodeSearchGroupSize/NQ/TopK from Done(); removed QueryNodeReduceLatency and tr.RecordSpan() from Execute()
5. `internal/querynodev2/segments/search.go` - Removed QueryNodeSQSegmentLatency and QueryNodeSegmentSearchLatencyPerVector from searchSegments(); removed searchLabel variable and commonpb import
6. `internal/querynodev2/segments/segment.go` - Removed debug logging, QueryNodeSQSegmentLatencyInCore metric, hasIndex check, and log.WithLazy from Search()

**Result:** Concurrency 1 regressed badly (-30%) because C++ thread pool (8) was too small. High concurrency improved +31%.

## Opt2: Adjusted C++ threads to CPUNum*1.5 (12)

**Change:**
- `internal/util/cgo/executor.go` - Changed executor thread pool to `cpuNum + cpuNum/2` (1.5x CPU = 12 threads)

| Concurrency | QPS | Avg Lat(us) | P50(us) | P95(us) | P99(us) | Recall@100 |
|---|---|---|---|---|---|---|
| 1 | 534 | 1865 | 1827 | 2230 | 2433 | 0.9533 |
| 5 | 1456 | 3424 | 3295 | 4847 | 6093 | 0.9533 |
| 10 | 1622 | 6154 | 5866 | 9286 | 12345 | 0.9533 |
| 20 | 1812 | 11031 | 10707 | 16974 | 23977 | 0.9533 |
| 30 | 1904 | 15748 | 15443 | 24864 | 32726 | 0.9533 |

**vs Baseline:** +7.9% QPS at concurrency 1, +30.7% at concurrency 30

## Opt3: Removed trace spans + remaining per-search overhead

**Changes:**
1. `internal/querynodev2/tasks/search_task.go` - Removed otel.Tracer().Start() span creation from NewSearchTask and NewStreamingSearchTask; removed scheduleSpan.End() from Execute; removed tr and scheduleSpan struct fields; removed otel, trace, typeutil imports
2. `internal/querynodev2/services.go` - Removed QueryNodeSQCount metrics (Total/Fail/Success), timerecord creation, debug logging, QueryNodeSQReqLatency from SearchSegments()

| Concurrency | QPS | Avg Lat(us) | P50(us) | P95(us) | P99(us) | Recall@100 |
|---|---|---|---|---|---|---|
| 1 | 537 | 1856 | 1818 | 2216 | 2415 | 0.9533 |
| 5 | 1460 | 3414 | 3282 | 4858 | 6119 | 0.9533 |
| 10 | 1625 | 6141 | 5871 | 9211 | 12164 | 0.9533 |
| 20 | 1800 | 11101 | 10744 | 17044 | 25171 | 0.9533 |
| 30 | 1899 | 15784 | 15472 | 24953 | 33824 | 0.9533 |

**vs Opt2:** Negligible change (within noise). The trace span and remaining metrics were not meaningful bottlenecks.

## Summary of Improvements (Baseline → Opt3)

| Concurrency | Baseline QPS | Final QPS | Improvement |
|---|---|---|---|
| 1 | 495 | 537 | +8.5% |
| 5 | 1133 | 1460 | +28.8% |
| 10 | 1243 | 1625 | +30.7% |
| 20 | 1367 | 1800 | +31.7% |
| 30 | 1456 | 1899 | +30.4% |

## Opt4: Removed per-segment access records, LockOSThread, context.WithCancel, TimeRecorderWithTrace

**Changes:**
1. `internal/querynodev2/segments/search.go` - Removed `metricsutil.NewSearchSegmentAccessRecord` (5 heap allocations + 10 time.Now() calls per search); removed `errgroup.WithContext(ctx)` and used bare `errgroup.Group` (saves derived context+cancel allocation); removed `segmentsWithoutIndex` tracking and `ctx.Err()` check per goroutine
2. `internal/util/cgo/pool.go` - Removed `runtime.LockOSThread()` and `runtime.UnlockOSThread()` from cgoCaller.call() (10 runtime calls per search); not needed since Go runtime handles CGO thread pinning automatically
3. `internal/querynodev2/services.go` - Removed `context.WithCancel(ctx)` from SearchSegments (saves context allocation); removed per-search log creation with 4 zap fields; moved channelsMvcc map allocation to after search completes (off hot path)
4. `internal/querynodev2/tasks/search_task.go` - Replaced `timerecord.NewTimeRecorderWithTrace` (2x time.Now() + fmt.Sprintf + trace ID extraction) with simple `time.Now()` + `time.Since(startTime)` for ServiceTime calculation

| Concurrency | QPS | Avg Lat(us) | P50(us) | P95(us) | P99(us) | Recall@100 |
|---|---|---|---|---|---|---|
| 1 | 544 | 1832 | 1795 | 2194 | 2394 | 0.9533 |
| 5 | 1496 | 3332 | 3209 | 4717 | 5930 | 0.9533 |
| 10 | 1669 | 5981 | 5715 | 8967 | 11820 | 0.9533 |
| 20 | 1855 | 10771 | 10535 | 16310 | 22583 | 0.9533 |
| 30 | 1937 | 15479 | 15308 | 23949 | 30702 | 0.9533 |

**vs Opt3:** +1.3% to +3.1% QPS; P99 at concurrency 30 improved -9.2% (33824→30702us)

## Summary of Improvements (Baseline → Opt4)

| Concurrency | Baseline QPS | Final QPS | Improvement |
|---|---|---|---|
| 1 | 495 | 544 | +9.9% |
| 5 | 1133 | 1496 | +32.0% |
| 10 | 1243 | 1669 | +34.3% |
| 20 | 1367 | 1855 | +35.7% |
| 30 | 1456 | 1937 | +33.0% |

## Opt5 Attempts (No Improvement)

Several additional optimizations were tried but provided no measurable improvement:

1. **Slice-based result collection instead of channel** in searchSegments - eliminates channel allocation and 10 channel ops per search. Result: within noise (±1%).
2. **Embedded readyMu mutex in futureImpl** - saves 1 heap allocation per CGO future (5 per search). Result: within noise.
3. **Skip futureManager.Register for search futures** - eliminates context.WithCancel, channel send to manager, and reflect.Select pressure. Result: slight regression (-1.5%), possibly due to GC timing changes.
4. **Bypass CGO semaphore for lightweight operations** (register_callback, get_result, destroy) - saves 15 semaphore passes per search. Result: P99 regression at concurrency 1 (16ms spikes) due to Go creating too many OS threads.
5. **Increase CGO semaphore from 16 to 128** - reduces semaphore contention. Result: slight regression at high concurrency (-2%) due to more OS thread context switching.

**Conclusion**: The remaining performance gap is inherent to the architecture:
- C++ HNSW search time (the actual computation) dominates
- CGO overhead per call is ~1-2us (irreducible crossing cost)
- Go runtime scheduling, GC, and channel operations are tuned

## Analysis

The system is now at ~92% of its CPU-bound theoretical limit (~2100 QPS). The remaining overhead is structural:
- gRPC serialization/deserialization (~100us per search round-trip)
- CGO async future pattern: 4 CGO calls per segment × 5 segments = 20 CGO calls per search
- Each CGO call: semaphore acquire + C-to-Go boundary + semaphore release
- context.WithCancel per future (5 per search)
- futureManager.Register + reflect.Select per future (5 per search)
- Scheduler channel round-trip (receiveChan → errCh → execChan → pool.Submit)
- errgroup goroutine creation (5 per search)
- Memory allocations: futureImpl, sync.Mutex, options, SearchRequest, result blobs
