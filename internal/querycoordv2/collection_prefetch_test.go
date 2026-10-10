package querycoordv2

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/mocks/streamingcoord/server/mock_broadcaster"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster/broadcast"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/internal/views/coord/balancer"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	streamingtypes "github.com/milvus-io/milvus/pkg/v3/streaming/util/types"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestPrefetchSmallSegmentsScopeAndAdmission(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	valid := func(path string) *querypb.SegmentLoadInfo {
		return &querypb.SegmentLoadInfo{StorageVersion: storage.StorageV3, ManifestPath: path, NumOfRows: 1000,
			IndexInfos: []*querypb.FieldIndexInfo{{EnableIndex: true}}}
	}
	oversized := valid("large")
	oversized.NumOfRows = 4097
	indexed := valid("indexed")
	indexed.IndexInfos = []*querypb.FieldIndexInfo{{IndexFilePaths: []string{"index-object"}}}
	legacy := valid("legacy")
	legacy.StorageVersion = storage.StorageV2
	var starts, ends int
	end := prefetchSmallSegments(ctx, []*querypb.SegmentLoadInfo{
		nil, oversized, indexed, legacy, valid("one"), valid("two"),
	}, []int64{0, 1, 100, 101}, func(path string, fields []int64) func() (uint64, uint64) {
		require.Equal(t, "one", path)
		require.Equal(t, []int64{0, 1, 100, 101}, fields)
		starts++
		cancel() // Cancellation between segments prevents additional speculative work.
		return func() (uint64, uint64) { ends++; return 2, 0 }
	})
	require.Equal(t, 1, starts)
	require.Zero(t, ends, "scope must stay alive through WAL and readiness")
	end()
	end()
	require.Equal(t, 1, ends, "all exits must release each scope exactly once")
}

type prefetchDataProvider struct{ emptyDataViewProvider }

func (prefetchDataProvider) DataViewSnapshotForCollections(context.Context, map[int64]struct{}) *balancer.DataViewSnapshot {
	return balancer.NewDataViewSnapshot(1, []*viewpb.DataViewOfCollection{{CollectionId: 100,
		Shards: []*viewpb.DataViewOfShard{{Partitions: []*viewpb.DataViewOfPartition{{SegmentIds: []int64{200}}}}},
	}}, nil)
}

type prefetchMixCoord struct{ types.MixCoord }

func (prefetchMixCoord) DataViewProvider() balancer.DataViewProvider { return prefetchDataProvider{} }

func TestAutoLoadPrefetchDoesNotBypassWALFailure(t *testing.T) {
	params := paramtable.Get()
	require.NoError(t, params.Save(params.CommonCfg.AutoLoadFilePrefetchEnabled.Key, "true"))
	t.Cleanup(func() { _ = params.Reset(params.CommonCfg.AutoLoadFilePrefetchEnabled.Key) })
	s, _, broker := newLoadConfigQViewsServer(t)
	s.ctx = context.Background()
	s.UpdateStateCode(commonpb.StateCode_Healthy)
	t.Cleanup(s.qviewsRuntime.stop)
	const channel = "by-dev-rootcoord-dml_100v0"
	coll := testDescribeCollection(100, []string{channel})
	broker.EXPECT().DescribeCollection(mock.Anything, int64(100)).Return(coll, nil).Times(3)
	broker.EXPECT().GetPartitions(mock.Anything, int64(100)).Return([]int64{10}, nil).Once()
	broker.EXPECT().GetCollectionLoadInfo(mock.Anything, int64(100)).Return([]string{meta.DefaultResourceGroupName}, int64(1), nil).Once()
	mix := mocks.NewMixCoord(t)
	s.mixCoord = prefetchMixCoord{mix}
	mix.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&indexpb.DescribeIndexResponse{Status: merr.Success()}, nil).Once()
	mix.EXPECT().GetQueryViewSegmentLoadInfos(mock.Anything, int64(100), []int64{200}).Return(
		[]*querypb.SegmentLoadInfo{{StorageVersion: storage.StorageV3, ManifestPath: "immutable-manifest", NumOfRows: 1000}}, nil, nil).Once()
	var started, closed atomic.Int32
	prepareStarted := make(chan struct{})
	patch := mockey.Mock(segcore.BeginLoadFilePrefetch).To(func(path string, fields []int64) func() (uint64, uint64) {
		started.Add(1)
		close(prepareStarted)
		return func() (uint64, uint64) { closed.Add(1); return 2, 0 }
	}).Build()
	t.Cleanup(func() { patch.UnPatch() })
	bapi := mock_broadcaster.NewMockBroadcastAPI(t)
	bapi.EXPECT().Broadcast(mock.Anything, mock.Anything).RunAndReturn(
		func(context.Context, message.BroadcastMutableMessage) (*streamingtypes.BroadcastAppendResult, error) {
			select {
			case <-prepareStarted:
			case <-time.After(time.Second):
				t.Error("prefetch did not run while WAL broadcast was blocked")
				return nil, merr.ErrServiceUnavailable
			}
			require.EqualValues(t, 1, started.Load(), "prefetch should overlap WAL broadcast")
			require.Zero(t, closed.Load(), "scope must remain alive during broadcast")
			require.False(t, s.qviewsRuntime.loadConfigStore.Contains(100))
			return nil, merr.ErrServiceUnavailable
		}).Once()
	bapi.EXPECT().Close().Return().Once()
	bc := mock_broadcaster.NewMockBroadcaster(t)
	bc.EXPECT().Close().Return().Maybe()
	bc.EXPECT().WithResourceKeys(mock.Anything, mock.Anything, mock.Anything).Return(bapi, nil).Once()
	broadcast.ResetBroadcaster()
	broadcast.Register(bc)
	t.Cleanup(broadcast.ResetBroadcaster)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	status, err := s.EnsureCollectionReady(ctx, &querypb.EnsureCollectionReadyRequest{
		CollectionID: 100, ExpectedVchannels: []string{channel},
	})
	require.NoError(t, err)
	require.ErrorIs(t, merr.Error(status), merr.ErrServiceUnavailable)
	require.EqualValues(t, 1, closed.Load(), "WAL failure must discard prepared data")
	require.False(t, s.qviewsRuntime.loadConfigStore.Contains(100), "prefetch may not publish a load config")
}

func TestScopedPrefetchCleanupJoinsLatePreparation(t *testing.T) {
	started := make(chan struct{})
	finish := make(chan struct{})
	var closed atomic.Int32
	end := startScopedPrefetch(context.Background(), func(ctx context.Context) func() {
		close(started)
		<-ctx.Done()
		<-finish
		return func() { closed.Add(1) }
	})
	<-started
	done := make(chan struct{})
	go func() { end(); close(done) }()
	select {
	case <-done:
		t.Fatal("cleanup returned before late preparation was joined")
	case <-time.After(20 * time.Millisecond):
	}
	close(finish)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cleanup did not join preparation")
	}
	end()
	require.EqualValues(t, 1, closed.Load())
}
