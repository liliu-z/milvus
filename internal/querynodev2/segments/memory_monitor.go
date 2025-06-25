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

package segments

import (
	"context"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/util/hardware"
)

// MemoryMonitor monitors physical memory usage with periodic updates
type MemoryMonitor struct {
	usedMemory    *atomic.Uint64
	updateCtx     context.Context
	updateCancel  context.CancelFunc
	updateDone    chan struct{}
	updateInterval time.Duration
}

// NewMemoryMonitor creates a new memory monitor
func NewMemoryMonitor() *MemoryMonitor {
	monitor := &MemoryMonitor{
		usedMemory:     atomic.NewUint64(0),
		updateInterval: 1 * time.Second, // Update every second
	}
	
	// Initialize with current memory usage
	monitor.usedMemory.Store(hardware.GetUsedMemoryCount())
	
	// Start monitoring loop
	monitor.startMonitoring()
	
	return monitor
}

// NewMemoryMonitorWithInterval creates a new memory monitor with custom update interval
func NewMemoryMonitorWithInterval(interval time.Duration) *MemoryMonitor {
	monitor := &MemoryMonitor{
		usedMemory:     atomic.NewUint64(0),
		updateInterval: interval,
	}
	
	// Initialize with current memory usage
	monitor.usedMemory.Store(hardware.GetUsedMemoryCount())
	
	// Start monitoring loop
	monitor.startMonitoring()
	
	return monitor
}

// GetUsedMemory returns the current used memory count
func (m *MemoryMonitor) GetUsedMemory() uint64 {
	return m.usedMemory.Load()
}

// ForceUpdate forces an immediate update of memory usage
func (m *MemoryMonitor) ForceUpdate() {
	usedMemory := hardware.GetUsedMemoryCount()
	m.usedMemory.Store(usedMemory)
	log.Debug("memory usage updated",
		zap.Uint64("usedMemory", usedMemory),
	)
}

// startMonitoring starts the background memory monitoring loop
func (m *MemoryMonitor) startMonitoring() {
	m.updateCtx, m.updateCancel = context.WithCancel(context.Background())
	m.updateDone = make(chan struct{})

	go func() {
		defer close(m.updateDone)
		ticker := time.NewTicker(m.updateInterval)
		defer ticker.Stop()

		for {
			select {
			case <-m.updateCtx.Done():
				return
			case <-ticker.C:
				m.ForceUpdate()
			}
		}
	}()
}

// Close stops the memory monitor
func (m *MemoryMonitor) Close() {
	if m.updateCancel != nil {
		m.updateCancel()
		<-m.updateDone
	}
} 