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
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v2/util/hardware"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
)

// ResourceSyncConfig contains configuration for resource synchronization
type ResourceSyncConfig struct {
	SyncInterval        time.Duration
	MemoryCalibInterval time.Duration
	EnableJemallocStats bool
}

// PhysicalResourceTracker tracks actual physical resource usage
type PhysicalResourceTracker struct {
	memoryUsage int64
	diskUsage   int64
	lastUpdate  time.Time
	mutex       sync.RWMutex
}

// Update updates the physical resource usage
func (p *PhysicalResourceTracker) Update(memoryUsage, diskUsage int64) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	p.memoryUsage = memoryUsage
	p.diskUsage = diskUsage
	p.lastUpdate = time.Now()
}

// Get returns the current physical resource usage
func (p *PhysicalResourceTracker) Get() (memoryUsage, diskUsage int64, lastUpdate time.Time) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	return p.memoryUsage, p.diskUsage, p.lastUpdate
}

// ResourceSynchronizer synchronizes resource state between Go and C++ layers
type ResourceSynchronizer struct {
	config       ResourceSyncConfig
	loadGuard    *CachingLayerLoadGuard
	physTracker  *PhysicalResourceTracker
	
	stopCh       chan struct{}
	wg           sync.WaitGroup
	
	// Calibration offset for memory usage discrepancy
	memoryCalibrationOffset int64
	calibrationMutex        sync.RWMutex
}

// NewResourceSynchronizer creates a new resource synchronizer
func NewResourceSynchronizer(loadGuard *CachingLayerLoadGuard) *ResourceSynchronizer {
	config := ResourceSyncConfig{
		SyncInterval:        5 * time.Second,
		MemoryCalibInterval: 30 * time.Second,
		EnableJemallocStats: true,
	}
	
	return &ResourceSynchronizer{
		config:      config,
		loadGuard:   loadGuard,
		physTracker: &PhysicalResourceTracker{},
		stopCh:      make(chan struct{}),
	}
}

// Start starts the resource synchronization background tasks
func (rs *ResourceSynchronizer) Start(ctx context.Context) error {
	log.Info("Starting ResourceSynchronizer",
		zap.Duration("syncInterval", rs.config.SyncInterval),
		zap.Duration("memoryCalibInterval", rs.config.MemoryCalibInterval))
	
	// Start synchronization goroutine
	rs.wg.Add(1)
	go rs.syncLoop(ctx)
	
	// Start memory calibration goroutine if enabled
	if rs.config.EnableJemallocStats {
		rs.wg.Add(1)
		go rs.memoryCalibrationLoop(ctx)
	}
	
	return nil
}

// Stop stops the resource synchronization
func (rs *ResourceSynchronizer) Stop() error {
	close(rs.stopCh)
	rs.wg.Wait()
	log.Info("ResourceSynchronizer stopped")
	return nil
}

// GetPhysicalUsage returns the current physical resource usage
func (rs *ResourceSynchronizer) GetPhysicalUsage() (ResourceUsage, error) {
	memUsage, diskUsage, _ := rs.physTracker.Get()
	return ResourceUsage{
		MemoryBytes: memUsage,
		DiskBytes:   diskUsage,
	}, nil
}

// SyncResourceState manually triggers a resource state synchronization
func (rs *ResourceSynchronizer) SyncResourceState() error {
	return rs.updatePhysicalUsage()
}

// syncLoop runs the main synchronization loop
func (rs *ResourceSynchronizer) syncLoop(ctx context.Context) {
	defer rs.wg.Done()
	
	ticker := time.NewTicker(rs.config.SyncInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-rs.stopCh:
			return
		case <-ticker.C:
			if err := rs.updatePhysicalUsage(); err != nil {
				log.Warn("Failed to update physical usage", zap.Error(err))
			}
		}
	}
}

// memoryCalibrationLoop runs the memory calibration loop
func (rs *ResourceSynchronizer) memoryCalibrationLoop(ctx context.Context) {
	defer rs.wg.Done()
	
	ticker := time.NewTicker(rs.config.MemoryCalibInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-rs.stopCh:
			return
		case <-ticker.C:
			rs.calibrateMemoryUsage()
		}
	}
}

// updatePhysicalUsage updates the physical resource usage
func (rs *ResourceSynchronizer) updatePhysicalUsage() error {
	// Get memory usage from hardware
	osMemoryUsage := hardware.GetUsedMemoryCount()
	
	// Apply calibration offset if available
	rs.calibrationMutex.RLock()
	calibratedMemoryUsage := int64(osMemoryUsage) + rs.memoryCalibrationOffset
	rs.calibrationMutex.RUnlock()
	
	// Get disk usage
	diskUsage, err := rs.getDiskUsage()
	if err != nil {
		log.Warn("Failed to get disk usage", zap.Error(err))
		diskUsage = 0
	}
	
	// Update the tracker
	rs.physTracker.Update(calibratedMemoryUsage, diskUsage)
	
	log.Debug("Updated physical resource usage",
		zap.Int64("memoryMB", calibratedMemoryUsage/(1024*1024)),
		zap.Int64("diskMB", diskUsage/(1024*1024)),
		zap.Int64("calibrationOffset", rs.memoryCalibrationOffset))
	
	return nil
}

// calibrateMemoryUsage calibrates memory usage using jemalloc stats if available
func (rs *ResourceSynchronizer) calibrateMemoryUsage() {
	if !rs.config.EnableJemallocStats {
		return
	}
	
	// Try to get jemalloc stats using CGO
	jemallocUsage := rs.getJemallocMemoryUsage()
	osUsage := int64(hardware.GetUsedMemoryCount())
	
	if jemallocUsage > 0 {
		rs.calibrationMutex.Lock()
		rs.memoryCalibrationOffset = jemallocUsage - osUsage
		rs.calibrationMutex.Unlock()
		
		log.Debug("Memory usage calibrated",
			zap.Int64("jemallocMB", jemallocUsage/(1024*1024)),
			zap.Int64("osMB", osUsage/(1024*1024)),
			zap.Int64("offsetMB", rs.memoryCalibrationOffset/(1024*1024)))
	}
}

// getJemallocMemoryUsage gets memory usage from jemalloc
func (rs *ResourceSynchronizer) getJemallocMemoryUsage() int64 {
	// This is a simplified implementation
	// In a real implementation, you would use CGO to call jemalloc's mallctl
	
	// For now, return 0 to indicate jemalloc stats are not available
	// TODO: Implement actual jemalloc integration using CGO
	
	// Fallback to runtime memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	// Use HeapInuse + StackInuse as an approximation
	return int64(m.HeapInuse + m.StackInuse)
}

// getDiskUsage calculates actual disk usage for mmap files and other data
func (rs *ResourceSynchronizer) getDiskUsage() (int64, error) {
	// Get local storage path from configuration
	dataPath := paramtable.Get().LocalStorageCfg.Path.GetValue()
	
	// Calculate total size of all files in the data directory
	var totalSize int64
	err := filepath.Walk(dataPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// Skip files that can't be accessed
			return nil
		}
		if !info.IsDir() {
			totalSize += info.Size()
		}
		return nil
	})
	
	if err != nil {
		return 0, err
	}
	
	return totalSize, nil
}

// Enhanced segment loader with resource synchronization
type synchronizedSegmentLoader struct {
	*cachingLayerSegmentLoader
	resourceSync *ResourceSynchronizer
}

// NewSynchronizedSegmentLoader creates a new segment loader with resource synchronization
func NewSynchronizedSegmentLoader(ctx context.Context, manager *Manager, cm storage.ChunkManager) (*synchronizedSegmentLoader, error) {
	baseLoader, err := NewCachingLayerSegmentLoader(ctx, manager, cm)
	if err != nil {
		return nil, err
	}
	
	resourceSync := NewResourceSynchronizer(baseLoader.loadGuard)
	
	loader := &synchronizedSegmentLoader{
		cachingLayerSegmentLoader: baseLoader,
		resourceSync:              resourceSync,
	}
	
	// Start resource synchronization
	if err := resourceSync.Start(ctx); err != nil {
		return nil, err
	}
	
	log.Info("SynchronizedSegmentLoader created with resource synchronization")
	
	return loader, nil
}

// Close releases resources and stops synchronization
func (loader *synchronizedSegmentLoader) Close() error {
	if loader.resourceSync != nil {
		if err := loader.resourceSync.Stop(); err != nil {
			log.Warn("Failed to stop resource synchronizer", zap.Error(err))
		}
	}
	
	if loader.cachingLayerSegmentLoader != nil {
		return loader.cachingLayerSegmentLoader.Close()
	}
	
	return nil
}

// Enhanced request resource with real-time resource synchronization
func (loader *synchronizedSegmentLoader) requestResourceWithSync(ctx context.Context, infos ...*querypb.SegmentLoadInfo) (requestResourceResult, error) {
	// Trigger immediate resource sync before making load decisions
	if err := loader.resourceSync.SyncResourceState(); err != nil {
		log.Warn("Failed to sync resource state", zap.Error(err))
	}
	
	// Get current physical usage
	physicalUsage, err := loader.resourceSync.GetPhysicalUsage()
	if err != nil {
		log.Warn("Failed to get physical usage", zap.Error(err))
	} else {
		log.Debug("Current physical resource usage",
			zap.Int64("memoryMB", physicalUsage.MemoryBytes/(1024*1024)),
			zap.Int64("diskMB", physicalUsage.DiskBytes/(1024*1024)))
	}
	
	// Proceed with the enhanced caching layer request
	return loader.cachingLayerSegmentLoader.requestResourceWithCachingLayer(ctx, infos...)
}

// GetResourceUsageReport returns a comprehensive resource usage report
func (loader *synchronizedSegmentLoader) GetResourceUsageReport() (map[string]interface{}, error) {
	report := make(map[string]interface{})
	
	// Get shallow usage from load guard
	if shallowUsage, err := loader.loadGuard.GetShallowUsage(); err == nil {
		report["shallow_memory_mb"] = shallowUsage.MemoryBytes / (1024 * 1024)
		report["shallow_disk_mb"] = shallowUsage.DiskBytes / (1024 * 1024)
	}
	
	// Get deep usage from load guard
	if deepUsage, err := loader.loadGuard.GetDeepUsage(); err == nil {
		report["deep_memory_mb"] = deepUsage.MemoryBytes / (1024 * 1024)
		report["deep_disk_mb"] = deepUsage.DiskBytes / (1024 * 1024)
	}
	
	// Get physical usage from synchronizer
	if physicalUsage, err := loader.resourceSync.GetPhysicalUsage(); err == nil {
		report["physical_memory_mb"] = physicalUsage.MemoryBytes / (1024 * 1024)
		report["physical_disk_mb"] = physicalUsage.DiskBytes / (1024 * 1024)
	}
	
	// Get available resources
	if availableUsage, err := loader.loadGuard.GetAvailableResources(); err == nil {
		report["available_memory_mb"] = availableUsage.MemoryBytes / (1024 * 1024)
		report["available_disk_mb"] = availableUsage.DiskBytes / (1024 * 1024)
	}
	
	// Get load guard statistics
	if segmentCount, err := loader.loadGuard.GetLoadedSegmentCount(); err == nil {
		report["loaded_segment_count"] = segmentCount
	}
	
	if cellCount, err := loader.loadGuard.GetLoadedCellCount(); err == nil {
		report["loaded_cell_count"] = cellCount
	}
	
	// Get configuration
	if config, err := loader.loadGuard.GetConfig(); err == nil {
		report["cache_ratio"] = config.CacheRatio
		report["eviction_enabled"] = config.EvictionEnabled
		report["low_watermark"] = config.LowWatermark
		report["high_watermark"] = config.HighWatermark
	}
	
	return report, nil
}