package cgo

import (
	"github.com/milvus-io/milvus/pkg/v2/util/hardware"
)

var caller *cgoCaller

func initCaller(nodeID string) {
	// Large semaphore: AsyncSearch is non-blocking (just queues to C++ executor),
	// so the semaphore only needs to prevent excessive simultaneous CGO boundary crossings.
	chSize := int64(hardware.GetCPUNum()) * 4
	if chSize <= 0 {
		chSize = 1
	}
	caller = &cgoCaller{
		ch: make(chan struct{}, chSize),
	}
}

// getCGOCaller returns the cgoCaller instance.
func getCGOCaller() *cgoCaller {
	return caller
}

// cgoCaller is a limiter to restrict the number of concurrent cgo calls.
type cgoCaller struct {
	ch chan struct{}
}

// call calls the work function with a semaphore to restrict the number of concurrent cgo calls.
func (c *cgoCaller) call(name string, work func()) {
	c.ch <- struct{}{}
	work()
	<-c.ch
}

// callLight calls lightweight CGO functions (callbacks, get result, destroy) without the semaphore.
// These operations are <1us and don't need concurrency limiting.
func (c *cgoCaller) callLight(work func()) {
	work()
}
