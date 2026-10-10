package cgo

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func withNativeFutureWaiter(t *testing.T) {
	previous := nativeFutureWaitSlots
	nativeFutureWaitSlots = make(chan struct{}, 1)
	t.Cleanup(func() {
		require.Empty(t, nativeFutureWaitSlots)
		nativeFutureWaitSlots = previous
	})
}

// Reuse the existing behavioral contract, including native errors, cancellation,
// deadlines, non-cooperative work, concurrent consumption, and release.
func TestNativeFutureContract(t *testing.T) {
	withNativeFutureWaiter(t)
	t.Run("success_and_double_consume", TestFutureWithSuccessCase)
	t.Run("non_cooperative_cancel", TestFutureWithCaseNoInterrupt)
	t.Run("native_errors_and_context_cancel", TestFutures)
	t.Run("field_not_loaded_status", TestFutureFieldNotLoadedIsRetriable)
	t.Run("concurrent_futures", TestConcurrent)
	t.Run("release_cancel_race", TestFutureWithConcurrentReleaseAndCancel)
}

func TestNativeFutureSaturationStillCancels(t *testing.T) {
	withNativeFutureWaiter(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	first := createFutureWithTestCase(ctx, testCase{
		interval: time.Millisecond, loopCnt: 10000, caseNo: 100,
	})
	defer first.Release()
	done := make(chan error, 2)
	go func() {
		_, err := first.BlockAndLeakyGet()
		done <- err
	}()
	require.Eventually(t, func() bool { return len(nativeFutureWaitSlots) == 1 }, time.Second, time.Millisecond)
	// Native wait must not hold a regular CGO slot needed by future_cancel.
	require.Empty(t, getCGOCaller().ch)
	callbackCount := func() uint64 {
		var metric dto.Metric
		hist := metrics.CGODuration.WithLabelValues(getCGOCaller().nodeID, "future_go_register_ready_callback")
		require.NoError(t, hist.(prometheus.Metric).Write(&metric))
		return metric.GetHistogram().GetSampleCount()
	}
	before := callbackCount()
	second := createFutureWithTestCase(ctx, testCase{
		interval: time.Millisecond, loopCnt: 10000, caseNo: 100,
	})
	defer second.Release()
	go func() {
		_, err := second.BlockAndLeakyGet()
		done <- err
	}()
	require.Eventually(t, func() bool { return callbackCount() > before }, time.Second, time.Millisecond)
	cancel()
	for range 2 {
		select {
		case err := <-done:
			require.ErrorIs(t, err, merr.ErrSegcoreFollyCancel)
			require.True(t, errors.Is(err, context.Canceled))
		case <-time.After(2 * time.Second):
			t.Fatal("native wait saturation blocked cancellation")
		}
	}
}

func TestNativeFutureConcurrentWaiters(t *testing.T) {
	withNativeFutureWaiter(t)
	future := createFutureWithTestCase(context.Background(), testCase{
		interval: time.Millisecond, loopCnt: 100, caseNo: 100,
	})
	defer future.Release()
	var wg sync.WaitGroup
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			future.BlockUntilReady()
		}()
	}
	wg.Wait()
	result, err := future.BlockAndLeakyGet()
	require.NoError(t, err)
	require.Equal(t, 100, getCInt(result))
	freeCInt(result)
	_, err = future.BlockAndLeakyGet()
	require.ErrorIs(t, err, merr.ErrServiceInternal)
}
