// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package observers

import (
	"context"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
	"go.uber.org/atomic"

	"github.com/milvus-io/milvus/internal/util/proxyutil"
	"github.com/milvus-io/milvus/pkg/v3/proto/proxypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type LeaderCacheObserverTestSuite struct {
	suite.Suite

	mockProxyManager *proxyutil.MockProxyClientManager

	observer *LeaderCacheObserver
}

func (suite *LeaderCacheObserverTestSuite) SetupSuite() {
	paramtable.Init()
	suite.mockProxyManager = proxyutil.NewMockProxyClientManager(suite.T())
	suite.observer = NewLeaderCacheObserver(suite.mockProxyManager)
}

func (suite *LeaderCacheObserverTestSuite) TestInvalidateShardLeaderCache() {
	suite.observer.Start(context.TODO())
	defer suite.observer.Stop()

	ret := atomic.NewBool(false)
	collectionIDs := typeutil.NewConcurrentSet[int64]()
	suite.mockProxyManager.EXPECT().InvalidateShardLeaderCache(mock.Anything, mock.Anything).RunAndReturn(
		func(ctx context.Context, req *proxypb.InvalidateShardLeaderCacheRequest) error {
			collectionIDs.Upsert(req.GetCollectionIDs()...)
			collectionIDs := req.GetCollectionIDs()

			if len(collectionIDs) == 1 && lo.Contains(collectionIDs, 1) {
				ret.Store(true)
			}
			return nil
		})

	suite.observer.RegisterEvent(1)
	suite.Eventually(func() bool {
		return ret.Load()
	}, 3*time.Second, 1*time.Second)

	// test batch submit events
	collectionIDs = typeutil.NewConcurrentSet[int64]()
	suite.mockProxyManager.ExpectedCalls = nil
	suite.mockProxyManager.EXPECT().InvalidateShardLeaderCache(mock.Anything, mock.Anything).RunAndReturn(
		func(ctx context.Context, req *proxypb.InvalidateShardLeaderCacheRequest) error {
			collectionIDs.Upsert(req.GetCollectionIDs()...)
			return nil
		})
	suite.observer.RegisterEvent(1)
	suite.observer.RegisterEvent(2)
	suite.observer.RegisterEvent(3)
	suite.Eventually(func() bool {
		return collectionIDs.Contain(1) && collectionIDs.Contain(2) && collectionIDs.Contain(3)
	}, 3*time.Second, 1*time.Second)
}

// queryCoord.brokerTimeout is documented and read in milliseconds ("5000" = 5s) by the
// coordinator brokers. The InvalidateShardLeaderCache call issued from the observer's
// event loop must use the same unit; reading it as seconds turns 5s into 5000s.
func (suite *LeaderCacheObserverTestSuite) TestHandleEventUsesBrokerTimeoutInMilliseconds() {
	proxyManager := proxyutil.NewMockProxyClientManager(suite.T())
	observer := NewLeaderCacheObserver(proxyManager)

	remainingCh := make(chan time.Duration, 1)
	proxyManager.EXPECT().InvalidateShardLeaderCache(mock.Anything, mock.Anything).RunAndReturn(
		func(ctx context.Context, req *proxypb.InvalidateShardLeaderCacheRequest) error {
			deadline, ok := ctx.Deadline()
			suite.True(ok, "InvalidateShardLeaderCache must be called with a deadline")
			select {
			case remainingCh <- time.Until(deadline):
			default:
			}
			return nil
		}).Once()

	observer.Start(context.Background())
	defer observer.Stop()
	observer.RegisterEvent(1)

	var remaining time.Duration
	select {
	case remaining = <-remainingCh:
	case <-time.After(10 * time.Second):
		suite.FailNow("InvalidateShardLeaderCache was not called")
	}

	brokerTimeout := paramtable.Get().QueryCoordCfg.BrokerTimeout.GetAsDuration(time.Millisecond)
	suite.Greater(remaining, time.Duration(0))
	suite.LessOrEqual(remaining, brokerTimeout,
		"remaining %v exceeds queryCoord.brokerTimeout %v: the timeout is not read in milliseconds", remaining, brokerTimeout)
}

func TestLeaderCacheObserverTestSuite(t *testing.T) {
	suite.Run(t, new(LeaderCacheObserverTestSuite))
}
