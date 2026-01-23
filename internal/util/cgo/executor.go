package cgo

/*
#cgo pkg-config: milvus_core

#include "futures/future_c.h"
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
// Use CPU count + small headroom to reduce oversubscription while allowing
// single queries to fan out to multiple segments without queuing.
func initExecutor() {
	pt := paramtable.Get()
	cpuNum := hardware.GetCPUNum()
	initPoolSize := cpuNum + 1 // CPU+1: minimal headroom beyond CPU count
	if initPoolSize < 4 {
		initPoolSize = 4
	}
	C.executor_set_thread_num(C.int(initPoolSize))

	resetThreadNum := func(evt *config.Event) {
		if evt.HasUpdated {
			cpuNum := hardware.GetCPUNum()
			newSize := cpuNum + 1
			if newSize < 4 {
				newSize = 4
			}
			log.Info("reset cgo thread num", zap.Int("thread_num", newSize))
			C.executor_set_thread_num(C.int(newSize))
		}
	}
	pt.Watch(pt.QueryNodeCfg.MaxReadConcurrency.Key, config.NewHandler("cgo."+pt.QueryNodeCfg.MaxReadConcurrency.Key, resetThreadNum))
	pt.Watch(pt.QueryNodeCfg.CGOPoolSizeRatio.Key, config.NewHandler("cgo."+pt.QueryNodeCfg.CGOPoolSizeRatio.Key, resetThreadNum))
}
