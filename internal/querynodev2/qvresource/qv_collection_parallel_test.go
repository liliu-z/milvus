//go:build test && dynamic

package qvresource

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/qnview"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type concurrentMetadataProvider struct {
	describe func(context.Context) (*milvuspb.DescribeCollectionResponse, error)
	load     func(context.Context) (qnview.QueryViewLoadInfo, error)
}

func (p concurrentMetadataProvider) DescribeCollection(ctx context.Context, _ int64) (*milvuspb.DescribeCollectionResponse, error) {
	return p.describe(ctx)
}

func (p concurrentMetadataProvider) GetQueryViewLoadInfo(ctx context.Context, _ int64, _ qnview.QueryViewLoadInfoVersion) (qnview.QueryViewLoadInfo, error) {
	return p.load(ctx)
}

func parallelMetadataView() *qviews.QueryViewAtQueryNode {
	return qviews.NewQueryViewAtQueryNode(&viewpb.QueryViewMeta{CollectionId: 1, LoadInfoVersion: 7}, &viewpb.QueryViewOfQueryNode{}).(*qviews.QueryViewAtQueryNode)
}

func TestQueryViewMetadataRequestsOverlap(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	describeStarted, loadStarted := make(chan struct{}), make(chan struct{})
	provider := concurrentMetadataProvider{
		describe: func(ctx context.Context) (*milvuspb.DescribeCollectionResponse, error) {
			close(describeStarted)
			select {
			case <-loadStarted:
				return &milvuspb.DescribeCollectionResponse{Schema: &schemapb.CollectionSchema{Name: "c"}}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
		load: func(ctx context.Context) (qnview.QueryViewLoadInfo, error) {
			close(loadStarted)
			select {
			case <-describeStarted:
				return qnview.QueryViewLoadInfo{CollectionID: 1, Version: 7}, nil
			case <-ctx.Done():
				return qnview.QueryViewLoadInfo{}, ctx.Err()
			}
		},
	}
	collection := &fakeQVCollectionManager{}
	guard, _, err := newQueryViewCollectionRuntimeManager(provider, collection).Acquire(ctx, parallelMetadataView())
	require.NoError(t, err)
	require.NotNil(t, guard)
	require.Equal(t, 1, collection.putCount)
	guard.Release()
}

func TestQueryViewMetadataFailureCancelsAndJoinsLoadInfo(t *testing.T) {
	loadExited := make(chan struct{})
	provider := concurrentMetadataProvider{
		describe: func(context.Context) (*milvuspb.DescribeCollectionResponse, error) {
			return nil, merr.WrapErrCollectionNotFound(1)
		},
		load: func(ctx context.Context) (qnview.QueryViewLoadInfo, error) {
			defer close(loadExited)
			<-ctx.Done()
			return qnview.QueryViewLoadInfo{}, ctx.Err()
		},
	}
	collection := &fakeQVCollectionManager{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	guard, retryable, err := newQueryViewCollectionRuntimeManager(provider, collection).Acquire(ctx, parallelMetadataView())
	require.ErrorIs(t, err, merr.ErrCollectionNotFound)
	require.False(t, retryable)
	require.Nil(t, guard)
	require.Zero(t, collection.putCount)
	select {
	case <-loadExited:
	default:
		t.Fatal("Acquire returned with metadata work still running")
	}
}

func TestQueryViewParallelMetadataPreservesLoadInfoErrors(t *testing.T) {
	for _, test := range []struct {
		failure   error
		retryable bool
	}{
		{merr.WrapErrNodeNotMatch(1, 2), true},
		{merr.WrapErrCollectionNotFound(1), false},
		{context.DeadlineExceeded, true},
	} {
		provider := concurrentMetadataProvider{
			describe: func(context.Context) (*milvuspb.DescribeCollectionResponse, error) {
				return &milvuspb.DescribeCollectionResponse{Schema: &schemapb.CollectionSchema{Name: "c"}}, nil
			},
			load: func(context.Context) (qnview.QueryViewLoadInfo, error) {
				return qnview.QueryViewLoadInfo{}, test.failure
			},
		}
		collection := &fakeQVCollectionManager{}
		guard, retryable, err := newQueryViewCollectionRuntimeManager(provider, collection).Acquire(context.Background(), parallelMetadataView())
		require.ErrorIs(t, err, test.failure)
		require.Equal(t, test.retryable, retryable)
		require.Nil(t, guard)
		require.Zero(t, collection.putCount)
	}
}
