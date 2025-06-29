package snmanager

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/milvus-io/milvus/internal/mocks/streamingcoord/server/mock_balancer"
	"github.com/milvus-io/milvus/pkg/v2/streaming/util/types"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

type pChannelInfoAssigned struct {
	version   typeutil.VersionInt64Pair
	pchannels []types.PChannelInfoAssigned
}

func TestStreamingNodeManager(t *testing.T) {
	m := newStreamingNodeManager()
	b := mock_balancer.NewMockBalancer(t)

	ch := make(chan pChannelInfoAssigned, 1)
	b.EXPECT().WatchChannelAssignments(mock.Anything, mock.Anything).Run(
		func(ctx context.Context, cb func(typeutil.VersionInt64Pair, []types.PChannelInfoAssigned) error) {
			for {
				select {
				case <-ctx.Done():
					return
				case p := <-ch:
					cb(p.version, p.pchannels)
				}
			}
		})
	b.EXPECT().RegisterStreamingEnabledNotifier(mock.Anything).Return()
	m.SetBalancerReady(b)

	streamingNodes := m.GetStreamingQueryNodeIDs()
	assert.Empty(t, streamingNodes)

	ch <- pChannelInfoAssigned{
		version: typeutil.VersionInt64Pair{
			Global: 1,
			Local:  1,
		},
		pchannels: []types.PChannelInfoAssigned{
			{
				Channel: types.PChannelInfo{Name: "a_test", Term: 1},
				Node:    types.StreamingNodeInfo{ServerID: 1, Address: "localhost:1"},
			},
		},
	}

	listener := m.ListenNodeChanged()
	err := listener.Wait(context.Background())
	assert.NoError(t, err)

	node := m.GetWALLocated("a_test")
	assert.Equal(t, node, int64(1))
	streamingNodes = m.GetStreamingQueryNodeIDs()
	assert.Equal(t, len(streamingNodes), 1)

	assert.NoError(t, m.RegisterStreamingEnabledListener(context.Background(), NewStreamingReadyNotifier()))
}

func TestStreamingReadyNotifier(t *testing.T) {
	n := NewStreamingReadyNotifier()
	assert.False(t, n.IsReady())
	n.inner.Cancel()
	<-n.Ready()
	assert.True(t, n.IsReady())
	n.Release()
	assert.True(t, n.IsReady())
}
