package querycoordv2

import (
	"context"
	"sync"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

var (
	autoLoadPrefetchMetricsOnce sync.Once
	autoLoadPrefetchFiles       = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "milvus_coord_autoload_prefetch_files_total",
		Help: "Files reserved and consumed by scoped read-ahead during automatic loading.",
	}, []string{"state"})
	autoLoadPrefetchSegments = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "milvus_coord_autoload_prefetch_segments_total",
		Help: "Admission results for scoped automatic-load read-ahead.",
	}, []string{"result"})
)

// startAutoLoadFilePrefetch overlaps metadata preparation as well as file IO
// with WAL persistence. Cleanup joins preparation before discarding its scope,
// so a late preparation can never leave data behind after this operation ends.
func (s *Server) startAutoLoadFilePrefetch(ctx context.Context, coll *milvuspb.DescribeCollectionResponse) func() {
	if !paramtable.Get().CommonCfg.AutoLoadFilePrefetchEnabled.GetAsBool() {
		return func() {}
	}
	return startScopedPrefetch(ctx, func(ctx context.Context) func() { return s.beginAutoLoadFilePrefetch(ctx, coll) })
}

func startScopedPrefetch(ctx context.Context, prepare func(context.Context) func()) func() {
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan func(), 1)
	go func() { done <- prepare(ctx) }()
	var once sync.Once
	return func() {
		once.Do(func() {
			cancel()
			end := <-done
			end()
		})
	}
}

// beginAutoLoadFilePrefetch only prepares bytes. LoadCollection, WAL persistence,
// view publication and readiness retain their existing ordering and error path.
// The scope belongs to the shared load, never to an individual waiting request.
func (s *Server) beginAutoLoadFilePrefetch(ctx context.Context, coll *milvuspb.DescribeCollectionResponse) func() {
	noop := func() {}
	if !paramtable.Get().CommonCfg.AutoLoadFilePrefetchEnabled.GetAsBool() ||
		typeutil.IsExternalCollection(coll.GetSchema()) || ctx.Err() != nil {
		return noop
	}
	provider := (&mixCoordDataViewProvider{mixCoord: s.mixCoord}).provider()
	if provider == nil {
		return noop
	}
	id := coll.GetCollectionID()
	snapshot := provider.DataViewSnapshotForCollections(ctx, map[int64]struct{}{id: {}})
	ids := make([]int64, 0, 4)
	snapshot.RangeShards(id, func(shard *viewpb.DataViewOfShard) bool {
		for _, partition := range shard.GetPartitions() {
			for _, segmentID := range partition.GetSegmentIds() {
				ids = append(ids, segmentID)
				if len(ids) > 4 {
					return false
				}
			}
		}
		return true
	})
	if len(ids) == 0 || len(ids) > 4 {
		return noop
	}
	infos, _, err := s.mixCoord.GetQueryViewSegmentLoadInfos(ctx, id, ids)
	if err != nil || ctx.Err() != nil {
		return noop
	}
	fields := []int64{common.RowIDField, common.TimeStampField}
	for _, field := range typeutil.GetAllFieldSchemas(coll.GetSchema()) {
		if field.GetIsPrimaryKey() || typeutil.IsVectorType(field.GetDataType()) {
			fields = append(fields, field.GetFieldID())
		}
	}
	return prefetchSmallSegments(ctx, infos, fields, segcore.BeginLoadFilePrefetch)
}

func prefetchSmallSegments(ctx context.Context, infos []*querypb.SegmentLoadInfo, fields []int64,
	begin func(string, []int64) func() (uint64, uint64),
) func() {
	autoLoadPrefetchMetricsOnce.Do(func() { metrics.GetRegisterer().MustRegister(autoLoadPrefetchFiles, autoLoadPrefetchSegments) })
	ends := make([]func() (uint64, uint64), 0, len(infos))
	for _, info := range infos {
		if ctx.Err() != nil {
			break
		}
		reason := "eligible"
		switch {
		case info.GetStorageVersion() != storage.StorageV3:
			reason = "legacy"
		case info.GetManifestPath() == "":
			reason = "missing_manifest"
		case info.GetNumOfRows() <= 0 || info.GetNumOfRows() > 4096:
			reason = "rows"
		default:
			for _, index := range info.GetIndexInfos() {
				// NO_TRAIN indexes can have metadata without an index object.
				if len(index.GetIndexFilePaths()) != 0 {
					reason = "indexed"
					break
				}
			}
		}
		autoLoadPrefetchSegments.WithLabelValues(reason).Inc()
		if reason != "eligible" {
			continue
		}
		ends = append(ends, begin(info.GetManifestPath(), fields))
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			var reserved, consumed uint64
			for _, end := range ends {
				r, c := end()
				reserved += r
				consumed += c
			}
			autoLoadPrefetchFiles.WithLabelValues("reserved").Add(float64(reserved))
			autoLoadPrefetchFiles.WithLabelValues("consumed").Add(float64(consumed))
			mlog.Debug(ctx, "Auto-load file prefetch scope closed", mlog.Uint64("reserved", reserved), mlog.Uint64("consumed", consumed))
		})
	}
}
