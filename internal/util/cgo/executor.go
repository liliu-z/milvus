package cgo

/*
#cgo pkg-config: milvus_core

#include "futures/future_c.h"
#include <sys/prctl.h>

// Re-enable THP for this process. Go runtime disables THP via
// prctl(PR_SET_THP_DISABLE, 1) which prevents the C++ HNSW index
// from benefiting from huge pages. With 3GB+ of index data and random
// access patterns, TLB misses add ~50 cycles per vector access.
// THP eliminates this by using 2MB pages (fits in L2 STLB).
#include <stdio.h>
static void enable_thp_for_process() {
    int ret = prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0);
    fprintf(stderr, "[THP] prctl(PR_SET_THP_DISABLE, 0) = %d\n", ret);
}
*/
import "C"

import (
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/pkg/v2/config"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/util/hardware"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

// initExecutor initialize underlying cgo thread pool.
// Use 1x CPU count: with memory-latency-bound HNSW, fewer threads reduce L2 cache contention.
func init() {
	// Re-enable THP for this process. The Go runtime disables THP via
	// prctl(PR_SET_THP_DISABLE, 1) but C++ HNSW benefits from 2MB pages
	// to reduce TLB misses on 3GB+ index data with random access.
	C.enable_thp_for_process()
}

func initExecutor() {
	pt := paramtable.Get()
	cpuNum := hardware.GetCPUNum()
	initPoolSize := cpuNum * 3
	if initPoolSize < 8 {
		initPoolSize = 8
	}
	C.executor_set_thread_num(C.int(initPoolSize))

	resetThreadNum := func(evt *config.Event) {
		if evt.HasUpdated {
			cpuNum := hardware.GetCPUNum()
			newSize := cpuNum * 3
			if newSize < 8 {
				newSize = 8
			}
			log.Info("reset cgo thread num", zap.Int("thread_num", newSize))
			C.executor_set_thread_num(C.int(newSize))
		}
	}
	pt.Watch(pt.QueryNodeCfg.MaxReadConcurrency.Key, config.NewHandler("cgo."+pt.QueryNodeCfg.MaxReadConcurrency.Key, resetThreadNum))
	pt.Watch(pt.QueryNodeCfg.CGOPoolSizeRatio.Key, config.NewHandler("cgo."+pt.QueryNodeCfg.CGOPoolSizeRatio.Key, resetThreadNum))
}
