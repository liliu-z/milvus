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

/*
#cgo pkg-config: milvus_core

#include "cachinglayer/segment_load_guard_c.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v2/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util/hardware"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
	"github.com/milvus-io/milvus/pkg/v2/log"
)

// LoadGuardConfig wraps the C configuration structure
type LoadGuardConfig struct {
	OverloadPercentage   float64
	CacheRatio          float64
	EvictionEnabled     bool
	LowWatermark        float64
	HighWatermark       float64
	EvictionIntervalMs  int64
}

// ResourceUsage wraps the C resource usage structure
type ResourceUsage struct {
	MemoryBytes int64
	DiskBytes   int64
}

// SegmentInfo contains segment loading information
type SegmentInfo struct {
	SegmentUID      int64
	MetaSize        int64
	EvictableSize   int64
	InevictableSize int64
}

// CellInfo contains cell loading information
type CellInfo struct {
	SegmentUID int64
	CellCID    int64
	Size       int64
	Evictable  bool
}

// CachingLayerLoadGuard wraps the C++ SegmentLoadGuard for Go usage
type CachingLayerLoadGuard struct {
	guard C.CSegmentLoadGuard
	mutex sync.RWMutex
}

// NewCachingLayerLoadGuard creates a new load guard instance
func NewCachingLayerLoadGuard(config LoadGuardConfig) (*CachingLayerLoadGuard, error) {
	cConfig := C.CLoadGuardConfig{
		overload_percentage:   C.double(config.OverloadPercentage),
		cache_ratio:          C.double(config.CacheRatio), 
		eviction_enabled:     C.bool(config.EvictionEnabled),
		low_watermark:        C.double(config.LowWatermark),
		high_watermark:       C.double(config.HighWatermark),
		eviction_interval_ms: C.int64_t(config.EvictionIntervalMs),
	}

	guard := C.NewSegmentLoadGuard(cConfig)
	if guard == nil {
		errMsg := C.GoString(C.GetLastError())
		return nil, fmt.Errorf("failed to create SegmentLoadGuard: %s", errMsg)
	}

	loadGuard := &CachingLayerLoadGuard{
		guard: guard,
	}

	// Set finalizer to ensure cleanup
	runtime.SetFinalizer(loadGuard, (*CachingLayerLoadGuard).cleanup)

	return loadGuard, nil
}

// Close releases the C++ resources
func (g *CachingLayerLoadGuard) Close() error {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if g.guard != nil {
		C.DeleteSegmentLoadGuard(g.guard)
		g.guard = nil
		runtime.SetFinalizer(g, nil)
	}
	return nil
}

func (g *CachingLayerLoadGuard) cleanup() {
	g.Close()
}

// UpdateConfig updates the load guard configuration
func (g *CachingLayerLoadGuard) UpdateConfig(config LoadGuardConfig) error {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if g.guard == nil {
		return fmt.Errorf("load guard is closed")
	}

	cConfig := C.CLoadGuardConfig{
		overload_percentage:   C.double(config.OverloadPercentage),
		cache_ratio:          C.double(config.CacheRatio),
		eviction_enabled:     C.bool(config.EvictionEnabled),
		low_watermark:        C.double(config.LowWatermark),
		high_watermark:       C.double(config.HighWatermark),
		eviction_interval_ms: C.int64_t(config.EvictionIntervalMs),
	}

	C.UpdateLoadGuardConfig(g.guard, cConfig)
	return nil
}

// GetConfig retrieves current configuration
func (g *CachingLayerLoadGuard) GetConfig() (LoadGuardConfig, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return LoadGuardConfig{}, fmt.Errorf("load guard is closed")
	}

	cConfig := C.GetLoadGuardConfig(g.guard)
	return LoadGuardConfig{
		OverloadPercentage:   float64(cConfig.overload_percentage),
		CacheRatio:          float64(cConfig.cache_ratio),
		EvictionEnabled:     bool(cConfig.eviction_enabled),
		LowWatermark:        float64(cConfig.low_watermark),
		HighWatermark:       float64(cConfig.high_watermark),
		EvictionIntervalMs:  int64(cConfig.eviction_interval_ms),
	}, nil
}

// SetPhysicalLimits sets the physical resource limits
func (g *CachingLayerLoadGuard) SetPhysicalLimits(memoryLimit, diskLimit int64) error {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return fmt.Errorf("load guard is closed")
	}

	C.SetPhysicalLimits(g.guard, C.int64_t(memoryLimit), C.int64_t(diskLimit))
	return nil
}

// GetPhysicalLimits retrieves current physical limits
func (g *CachingLayerLoadGuard) GetPhysicalLimits() (memoryLimit, diskLimit int64, err error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return 0, 0, fmt.Errorf("load guard is closed")
	}

	usage := C.GetPhysicalLimits(g.guard)
	return int64(usage.memory_bytes), int64(usage.disk_bytes), nil
}

// CanLoadSegment checks if a segment can be loaded
func (g *CachingLayerLoadGuard) CanLoadSegment(info SegmentInfo) (bool, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return false, fmt.Errorf("load guard is closed")
	}

	cInfo := C.CSegmentInfo{
		segment_uid:      C.int64_t(info.SegmentUID),
		meta_size:       C.int64_t(info.MetaSize),
		evictable_size:  C.int64_t(info.EvictableSize),
		inevictable_size: C.int64_t(info.InevictableSize),
	}

	result := C.CanLoadSegment(g.guard, cInfo)
	return bool(result), nil
}

// LoadSegment loads a segment
func (g *CachingLayerLoadGuard) LoadSegment(info SegmentInfo) error {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return fmt.Errorf("load guard is closed")
	}

	cInfo := C.CSegmentInfo{
		segment_uid:      C.int64_t(info.SegmentUID),
		meta_size:       C.int64_t(info.MetaSize),
		evictable_size:  C.int64_t(info.EvictableSize),
		inevictable_size: C.int64_t(info.InevictableSize),
	}

	result := C.LoadSegment(g.guard, cInfo)
	if !bool(result) {
		errMsg := C.GoString(C.GetLastError())
		return fmt.Errorf("failed to load segment %d: %s", info.SegmentUID, errMsg)
	}
	return nil
}

// UnloadSegment unloads a segment
func (g *CachingLayerLoadGuard) UnloadSegment(segmentUID int64) error {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return fmt.Errorf("load guard is closed")
	}

	C.UnloadSegment(g.guard, C.int64_t(segmentUID))
	return nil
}

// CanLoadCell checks if a cell can be loaded
func (g *CachingLayerLoadGuard) CanLoadCell(info CellInfo) (bool, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return false, fmt.Errorf("load guard is closed")
	}

	cInfo := C.CCellInfo{
		segment_uid: C.int64_t(info.SegmentUID),
		cell_cid:   C.int64_t(info.CellCID),
		size:       C.int64_t(info.Size),
		evictable:  C.bool(info.Evictable),
	}

	result := C.CanLoadCell(g.guard, cInfo)
	return bool(result), nil
}

// LoadCell loads a cell
func (g *CachingLayerLoadGuard) LoadCell(info CellInfo) error {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return fmt.Errorf("load guard is closed")
	}

	cInfo := C.CCellInfo{
		segment_uid: C.int64_t(info.SegmentUID),
		cell_cid:   C.int64_t(info.CellCID),
		size:       C.int64_t(info.Size),
		evictable:  C.bool(info.Evictable),
	}

	result := C.LoadCell(g.guard, cInfo)
	if !bool(result) {
		errMsg := C.GoString(C.GetLastError())
		return fmt.Errorf("failed to load cell %d in segment %d: %s", info.CellCID, info.SegmentUID, errMsg)
	}
	return nil
}

// UnloadCell unloads a cell
func (g *CachingLayerLoadGuard) UnloadCell(segmentUID, cellCID int64) error {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return fmt.Errorf("load guard is closed")
	}

	C.UnloadCell(g.guard, C.int64_t(segmentUID), C.int64_t(cellCID))
	return nil
}

// GetShallowUsage returns the shallow resource usage
func (g *CachingLayerLoadGuard) GetShallowUsage() (ResourceUsage, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return ResourceUsage{}, fmt.Errorf("load guard is closed")
	}

	usage := C.GetShallowUsage(g.guard)
	return ResourceUsage{
		MemoryBytes: int64(usage.memory_bytes),
		DiskBytes:   int64(usage.disk_bytes),
	}, nil
}

// GetDeepUsage returns the deep resource usage
func (g *CachingLayerLoadGuard) GetDeepUsage() (ResourceUsage, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return ResourceUsage{}, fmt.Errorf("load guard is closed")
	}

	usage := C.GetDeepUsage(g.guard)
	return ResourceUsage{
		MemoryBytes: int64(usage.memory_bytes),
		DiskBytes:   int64(usage.disk_bytes),
	}, nil
}

// GetPhysicalUsage returns the current physical resource usage
func (g *CachingLayerLoadGuard) GetPhysicalUsage() (ResourceUsage, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return ResourceUsage{}, fmt.Errorf("load guard is closed")
	}

	usage := C.GetPhysicalUsage(g.guard)
	return ResourceUsage{
		MemoryBytes: int64(usage.memory_bytes),
		DiskBytes:   int64(usage.disk_bytes),
	}, nil
}

// GetAvailableResources returns available resources
func (g *CachingLayerLoadGuard) GetAvailableResources() (ResourceUsage, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return ResourceUsage{}, fmt.Errorf("load guard is closed")
	}

	usage := C.GetAvailableResources(g.guard)
	return ResourceUsage{
		MemoryBytes: int64(usage.memory_bytes),
		DiskBytes:   int64(usage.disk_bytes),
	}, nil
}

// TryEvict attempts to evict resources
func (g *CachingLayerLoadGuard) TryEvict(targetMemory, targetDisk int64) (bool, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return false, fmt.Errorf("load guard is closed")
	}

	result := C.TryEvict(g.guard, C.int64_t(targetMemory), C.int64_t(targetDisk))
	return bool(result), nil
}

// ForceEviction forces resource eviction
func (g *CachingLayerLoadGuard) ForceEviction() error {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return fmt.Errorf("load guard is closed")
	}

	C.ForceEviction(g.guard)
	return nil
}

// GetLoadedSegmentCount returns the number of loaded segments
func (g *CachingLayerLoadGuard) GetLoadedSegmentCount() (int64, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return 0, fmt.Errorf("load guard is closed")
	}

	count := C.GetLoadedSegmentCount(g.guard)
	return int64(count), nil
}

// GetLoadedCellCount returns the number of loaded cells
func (g *CachingLayerLoadGuard) GetLoadedCellCount() (int64, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if g.guard == nil {
		return 0, fmt.Errorf("load guard is closed")
	}

	count := C.GetLoadedCellCount(g.guard)
	return int64(count), nil
}

// Enhanced segment loader with caching layer integration
type cachingLayerSegmentLoader struct {
	*segmentLoader
	loadGuard *CachingLayerLoadGuard
}

// NewCachingLayerSegmentLoader creates a new segment loader with caching layer integration
func NewCachingLayerSegmentLoader(ctx context.Context, manager *Manager, cm storage.ChunkManager) (*cachingLayerSegmentLoader, error) {
	baseLoader := NewLoader(ctx, manager, cm)
	
	// Initialize load guard with default configuration from params
	config := LoadGuardConfig{
		OverloadPercentage: paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat(),
		CacheRatio:        paramtable.Get().QueryNodeCfg.CacheRatio.GetAsFloat(),
		EvictionEnabled:   paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool(),
		LowWatermark:     paramtable.Get().QueryNodeCfg.LoadGuardLowWatermark.GetAsFloat(),
		HighWatermark:    paramtable.Get().QueryNodeCfg.LoadGuardHighWatermark.GetAsFloat(),
		EvictionIntervalMs: paramtable.Get().QueryNodeCfg.LoadGuardEvictionIntervalMs.GetAsInt64(),
	}

	loadGuard, err := NewCachingLayerLoadGuard(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create load guard: %w", err)
	}

	// Set physical limits
	memoryLimit := int64(float64(hardware.GetMemoryCount()) * config.OverloadPercentage)
	diskLimit := int64(paramtable.Get().QueryNodeCfg.DiskCapacityLimit.GetAsUint64())
	
	if err := loadGuard.SetPhysicalLimits(memoryLimit, diskLimit); err != nil {
		loadGuard.Close()
		return nil, fmt.Errorf("failed to set physical limits: %w", err)
	}

	loader := &cachingLayerSegmentLoader{
		segmentLoader: baseLoader,
		loadGuard:     loadGuard,
	}

	log.Info("CachingLayerSegmentLoader created", 
		zap.Float64("cacheRatio", config.CacheRatio),
		zap.Bool("evictionEnabled", config.EvictionEnabled),
		zap.Int64("memoryLimitGB", memoryLimit/(1024*1024*1024)),
		zap.Int64("diskLimitGB", diskLimit/(1024*1024*1024)))

	return loader, nil
}

// Close releases resources
func (loader *cachingLayerSegmentLoader) Close() error {
	if loader.loadGuard != nil {
		return loader.loadGuard.Close()
	}
	return nil
}

// requestResource overrides the base loader's resource request logic
func (loader *cachingLayerSegmentLoader) requestResource(ctx context.Context, infos ...*querypb.SegmentLoadInfo) (requestResourceResult, error) {
	if len(infos) == 0 {
		return requestResourceResult{}, nil
	}

	segmentIDs := lo.Map(infos, func(info *querypb.SegmentLoadInfo, _ int) int64 {
		return info.GetSegmentID()
	})
	log := log.Ctx(ctx).With(zap.Int64s("segmentIDs", segmentIDs))

	// Convert load infos to segment infos for caching layer
	var totalEvictableSize, totalInevictableSize, totalMetaSize int64
	
	for _, info := range infos {
		// Get collection schema
		collection := loader.manager.Collection.Get(info.GetCollectionID())
		if collection == nil {
			return requestResourceResult{}, fmt.Errorf("collection %d not found", info.GetCollectionID())
		}
		
		// Calculate resource usage for this segment using existing logic
		factor := resourceEstimateFactor{
			memoryUsageFactor:        paramtable.Get().QueryNodeCfg.SegmentLoadMemoryUsageFactor.GetAsFloat(),
			memoryIndexUsageFactor:   paramtable.Get().QueryNodeCfg.IndexLoadMemoryUsageFactor.GetAsFloat(),
			mmapFieldEnableDiskFactor: paramtable.Get().QueryNodeCfg.MmapFieldEnableDiskFactor.GetAsFloat(),
			tempSegmentIndexFactor:    paramtable.Get().QueryNodeCfg.InterimIndexMemExpandRate.GetAsFloat(),
		}
		
		resourceUsage, err := getResourceUsageEstimateOfSegment(collection.Schema(), info, factor)
		if err != nil {
			return requestResourceResult{}, err
		}
		
		// Convert ResourceUsage to LoadResource
		loadResource := LoadResource{
			MemorySize: resourceUsage.MemorySize,
			DiskSize:   resourceUsage.DiskSize,
		}
		
		// Categorize resources based on data types
		evictableSize, inevictableSize, metaSize := categorizeResourceUsage(info, loadResource)
		
		totalEvictableSize += int64(evictableSize)
		totalInevictableSize += int64(inevictableSize)
		totalMetaSize += int64(metaSize)
	}

	// Check with caching layer load guard
	segmentInfo := SegmentInfo{
		SegmentUID:      segmentIDs[0], // Use first segment ID as representative
		MetaSize:        totalMetaSize,
		EvictableSize:   totalEvictableSize,
		InevictableSize: totalInevictableSize,
	}

	canLoad, err := loader.loadGuard.CanLoadSegment(segmentInfo)
	if err != nil {
		return requestResourceResult{}, fmt.Errorf("failed to check load admission: %w", err)
	}

	if !canLoad {
		// Try eviction if enabled
		config, err := loader.loadGuard.GetConfig()
		if err != nil {
			return requestResourceResult{}, err
		}
		
		if config.EvictionEnabled {
			// Calculate how much we need to evict
			available, err := loader.loadGuard.GetAvailableResources()
			if err != nil {
				return requestResourceResult{}, err
			}
			
			neededMemory := totalEvictableSize + totalInevictableSize - available.MemoryBytes
			if neededMemory > 0 {
				success, err := loader.loadGuard.TryEvict(neededMemory, 0)
				if err != nil {
					return requestResourceResult{}, err
				}
				if !success {
					return requestResourceResult{}, merr.WrapErrServiceMemoryLimitExceeded(
						float32(neededMemory + available.MemoryBytes), 
						float32(available.MemoryBytes))
				}
			}
		} else {
			return requestResourceResult{}, merr.WrapErrServiceMemoryLimitExceeded(
				float32(totalEvictableSize + totalInevictableSize), 
				float32(0)) // Placeholder
		}
	}

	// If we reach here, we can proceed with loading
	// Register segment with caching layer
	if err := loader.loadGuard.LoadSegment(segmentInfo); err != nil {
		return requestResourceResult{}, fmt.Errorf("failed to register segment load: %w", err)
	}

	// Return compatible result for existing code
	result := requestResourceResult{
		Resource: LoadResource{
			MemorySize: uint64(totalEvictableSize + totalInevictableSize),
			DiskSize:   0, // Disk handling would be added separately
		},
		CommittedResource: LoadResource{}, // This would be populated based on actual committed resources
		ConcurrencyLevel:  len(infos),
	}

	log.Info("resource request approved by caching layer", 
		zap.Int64("totalMemoryMB", (totalEvictableSize + totalInevictableSize)/(1024*1024)),
		zap.Int64("evictableMemoryMB", totalEvictableSize/(1024*1024)),
		zap.Int64("inevictableMemoryMB", totalInevictableSize/(1024*1024)))

	return result, nil
}

// categorizeResourceUsage categorizes resource usage into evictable, inevictable, and metadata
func categorizeResourceUsage(info *querypb.SegmentLoadInfo, resourceUsage LoadResource) (evictable, inevictable, meta uint64) {
	// Meta size is typically small - around 5% of total
	meta = resourceUsage.MemorySize / 20
	
	// Categorize based on data types and configuration
	var evictableMemory uint64
	var inevictableMemory uint64
	
	// Vector data is typically evictable if tiered storage is enabled
	for _, fieldBinlog := range info.GetBinlogPaths() {
		fieldID := fieldBinlog.GetFieldID()
		
		// Check if this field is configured as evictable
		// For now, assume vector fields are evictable, others are not
		// This should be enhanced based on actual field types and configuration
		if isVectorField(info.GetSchema(), fieldID) && paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool() {
			evictableMemory += estimateFieldMemory(fieldBinlog)
		} else {
			inevictableMemory += estimateFieldMemory(fieldBinlog)
		}
	}
	
	// Stats and delta logs are always inevictable
	for _, statslog := range info.GetStatslogs() {
		inevictableMemory += estimateFieldMemory(statslog)
	}
	
	for _, deltalog := range info.GetDeltalogs() {
		inevictableMemory += estimateFieldMemory(deltalog)
	}
	
	// Ensure we don't exceed total memory
	total := evictableMemory + inevictableMemory + meta
	if total > resourceUsage.MemorySize {
		// Scale down proportionally
		scale := float64(resourceUsage.MemorySize) / float64(total)
		evictableMemory = uint64(float64(evictableMemory) * scale)
		inevictableMemory = uint64(float64(inevictableMemory) * scale)
		meta = resourceUsage.MemorySize - evictableMemory - inevictableMemory
	}
	
	return evictableMemory, inevictableMemory, meta
}

// isVectorField checks if a field is a vector field
func isVectorField(schema *schemapb.CollectionSchema, fieldID int64) bool {
	if schema == nil {
		return false
	}
	
	for _, field := range schema.GetFields() {
		if field.GetFieldID() == fieldID {
			return typeutil.IsVectorType(field.GetDataType())
		}
	}
	
	return false
}

// estimateFieldMemory estimates memory usage for a field
func estimateFieldMemory(fieldBinlog *datapb.FieldBinlog) uint64 {
	var totalSize uint64
	for _, binlog := range fieldBinlog.GetBinlogs() {
		// Use memory_size if available, otherwise use log_size
		if binlog.GetMemorySize() > 0 {
			totalSize += uint64(binlog.GetMemorySize())
		} else {
			totalSize += binlog.GetLogSize()
		}
	}
	return totalSize
}