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
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/hardware"
)

// Cell represents a piece of data that can be loaded into memory/disk
type Cell struct {
	ID        string // unique identifier for the cell
	Size      uint64 // estimated size in bytes
	Evictable bool   // whether this cell can be evicted
	DeepLoaded *atomic.Bool // whether this cell is actually loaded
}

// NewCell creates a new cell
func NewCell(id string, size uint64, evictable bool) *Cell {
	return &Cell{
		ID:         id,
		Size:       size,
		Evictable:  evictable,
		DeepLoaded: atomic.NewBool(false),
	}
}

// IsDeepLoaded returns whether the cell is actually loaded
func (c *Cell) IsDeepLoaded() bool {
	return c.DeepLoaded.Load()
}

// SetDeepLoaded sets the deep loaded state
func (c *Cell) SetDeepLoaded(loaded bool) {
	c.DeepLoaded.Store(loaded)
}

// SegmentInfo represents segment resource information
type SegmentInfo struct {
	SegmentID       int64  // segment ID
	MetaSize        uint64 // metadata size, non-evictable
	EvictableSize   uint64 // total evictable data size
	InevictableSize uint64 // total non-evictable data size (excluding metadata)
	Cells           []*Cell // individual data cells
	IsShallowLoaded *atomic.Bool // whether segment is shallow loaded
}

// NewSegmentInfo creates a new segment info
func NewSegmentInfo(segmentID int64, metaSize uint64, cells []*Cell) *SegmentInfo {
	info := &SegmentInfo{
		SegmentID:       segmentID,
		MetaSize:        metaSize,
		Cells:           cells,
		IsShallowLoaded: atomic.NewBool(false),
	}
	
	// calculate evictable and inevictable sizes
	for _, cell := range cells {
		if cell.Evictable {
			info.EvictableSize += cell.Size
		} else {
			info.InevictableSize += cell.Size
		}
	}
	
	return info
}

// TotalSize returns the total size of the segment
func (s *SegmentInfo) TotalSize() uint64 {
	return s.MetaSize + s.EvictableSize + s.InevictableSize
}

// LoadGuardConfig contains configuration for the load guard
type LoadGuardConfig struct {
	OverloadPercentage float64 // maximum resource usage percentage
	CacheRatio         float64 // ratio of resources reserved for cache
	EvictionEnabled    bool    // whether eviction is enabled
	LowWatermark       float64 // low watermark for eviction
	HighWatermark      float64 // high watermark for eviction
	EvictionInterval   time.Duration // interval for eviction loop
}

// DefaultLoadGuardConfig returns default configuration
func DefaultLoadGuardConfig() LoadGuardConfig {
	return LoadGuardConfig{
		OverloadPercentage: paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat(),
		CacheRatio:         0.2,
		EvictionEnabled:    true,
		LowWatermark:       0.7,
		HighWatermark:      0.8,
		EvictionInterval:   3 * time.Second,
	}
}

// ValidateConfig validates the configuration
func (c *LoadGuardConfig) Validate() error {
	if c.CacheRatio <= 0 || c.CacheRatio > 1 {
		return fmt.Errorf("cache_ratio must be in (0, 1], got %f", c.CacheRatio)
	}
	if c.LowWatermark <= 0 || c.HighWatermark <= c.LowWatermark || c.OverloadPercentage <= c.HighWatermark {
		return fmt.Errorf("watermarks must be ordered: 0 < low(%f) < high(%f) < overload(%f)", 
			c.LowWatermark, c.HighWatermark, c.OverloadPercentage)
	}
	if c.OverloadPercentage > 1 {
		return fmt.Errorf("overload_percentage cannot exceed 1, got %f", c.OverloadPercentage)
	}
	return nil
}

// SegmentLoadGuard manages resource allocation and eviction for segments
type SegmentLoadGuard struct {
	config LoadGuardConfig
	limit  uint64 // total resource limit

	// Segment tracking - shallow loaded segments
	mu                 sync.RWMutex
	shallowEvictable   uint64 // total evictable size of shallow loaded segments
	shallowInevictable uint64 // total inevictable size of shallow loaded segments

	// Cell tracking - deep loaded data
	deepEvictable   *atomic.Uint64 // total evictable size of deep loaded data
	deepInevictable *atomic.Uint64 // total inevictable size of deep loaded data
	deepLoading     *atomic.Uint64 // total size of data currently being loaded

	// Eviction management
	evictionCtx    context.Context
	evictionCancel context.CancelFunc
	evictionDone   chan struct{}

	// Cell registry for eviction
	cellRegistry map[string]*Cell
	cellMutex    sync.RWMutex

	// Memory monitoring
	memoryMonitor *MemoryMonitor
}

// NewSegmentLoadGuard creates a new load guard
func NewSegmentLoadGuard(config LoadGuardConfig) (*SegmentLoadGuard, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	if !config.EvictionEnabled {
		config.CacheRatio = 1.0
	}

	guard := &SegmentLoadGuard{
		config:          config,
		limit:           uint64(float64(hardware.GetMemoryCount()) * config.OverloadPercentage),
		deepEvictable:   atomic.NewUint64(0),
		deepInevictable: atomic.NewUint64(0),
		deepLoading:     atomic.NewUint64(0),
		cellRegistry:    make(map[string]*Cell),
		memoryMonitor:   NewMemoryMonitor(),
	}

	if config.EvictionEnabled {
		guard.startEvictionLoop()
	}

	log.Info("SegmentLoadGuard created",
		zap.Float64("overloadPercentage", config.OverloadPercentage),
		zap.Float64("cacheRatio", config.CacheRatio),
		zap.Bool("evictionEnabled", config.EvictionEnabled),
		zap.Uint64("memoryLimit", guard.limit),
	)

	return guard, nil
}

// ShallowUsage returns the estimated resource usage for shallow loaded segments
func (g *SegmentLoadGuard) ShallowUsage() uint64 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.shallowInevictable + uint64(float64(g.shallowEvictable)*g.config.CacheRatio)
}

// DeepUsage returns the actual resource usage for deep loaded data
func (g *SegmentLoadGuard) DeepUsage() uint64 {
	return g.deepInevictable.Load() + g.deepEvictable.Load() + g.deepLoading.Load()
}

// SegmentLoadAdmission checks whether a segment can be shallow loaded
func (g *SegmentLoadGuard) SegmentLoadAdmission(ctx context.Context, segInfo *SegmentInfo) error {
	segShallowUsage := segInfo.MetaSize + segInfo.InevictableSize + 
		uint64(float64(segInfo.EvictableSize)*g.config.CacheRatio)
	
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check logical watermark
	currentShallowUsage := g.shallowInevictable + uint64(float64(g.shallowEvictable)*g.config.CacheRatio)
	if segShallowUsage+currentShallowUsage > g.limit {
		return fmt.Errorf("segment load rejected: shallow usage would exceed limit, "+
			"current=%d, requested=%d, limit=%d", 
			currentShallowUsage, segShallowUsage, g.limit)
	}

	// Check physical watermark
	physicalUsed := g.memoryMonitor.GetUsedMemory()
	if segInfo.MetaSize+physicalUsed > g.limit {
		return fmt.Errorf("segment load rejected: physical memory would exceed limit, "+
			"current=%d, requested=%d, limit=%d", 
			physicalUsed, segInfo.MetaSize, g.limit)
	}

	// Reserve resources
	g.shallowInevictable += segInfo.MetaSize + segInfo.InevictableSize
	g.shallowEvictable += segInfo.EvictableSize
	segInfo.IsShallowLoaded.Store(true)

	log.Debug("segment shallow load admitted",
		zap.Int64("segmentID", segInfo.SegmentID),
		zap.Uint64("metaSize", segInfo.MetaSize),
		zap.Uint64("evictableSize", segInfo.EvictableSize),
		zap.Uint64("inevictableSize", segInfo.InevictableSize),
		zap.Uint64("shallowUsage", currentShallowUsage+segShallowUsage),
	)

	return nil
}

// CellLoadAdmission checks whether a cell can be deep loaded
func (g *SegmentLoadGuard) CellLoadAdmission(ctx context.Context, cell *Cell) error {
	// Check if we have enough resources available
	logicalUsed := g.deepLoading.Load() + g.deepEvictable.Load() + g.deepInevictable.Load()
	physicalUsed := g.memoryMonitor.GetUsedMemory()
	used := max(logicalUsed, physicalUsed)
	
	available := int64(g.limit) - int64(used)
	needed := int64(cell.Size)

	if needed <= available || g.tryEvict(needed-available) {
		// Reserve resources for loading
		g.deepLoading.Add(cell.Size)
		
		// Register cell for eviction management
		g.cellMutex.Lock()
		g.cellRegistry[cell.ID] = cell
		g.cellMutex.Unlock()
		
		log.Debug("cell load admitted",
			zap.String("cellID", cell.ID),
			zap.Uint64("cellSize", cell.Size),
			zap.Bool("evictable", cell.Evictable),
			zap.Uint64("deepLoading", g.deepLoading.Load()),
		)
		
		return nil
	}

	return fmt.Errorf("cell load rejected: insufficient resources after eviction, "+
		"available=%d, needed=%d", available, needed)
}

// OnCellLoaded should be called when a cell is successfully loaded
func (g *SegmentLoadGuard) OnCellLoaded(cell *Cell) {
	g.deepLoading.Sub(cell.Size)
	if cell.Evictable {
		g.deepEvictable.Add(cell.Size)
	} else {
		g.deepInevictable.Add(cell.Size)
	}
	cell.SetDeepLoaded(true)
	
	log.Debug("cell loaded",
		zap.String("cellID", cell.ID),
		zap.Uint64("cellSize", cell.Size),
		zap.Bool("evictable", cell.Evictable),
		zap.Uint64("deepEvictable", g.deepEvictable.Load()),
		zap.Uint64("deepInevictable", g.deepInevictable.Load()),
	)
}

// OnCellEvicted should be called when a cell is evicted or released
func (g *SegmentLoadGuard) OnCellEvicted(cell *Cell) {
	if cell.Evictable {
		g.deepEvictable.Sub(cell.Size)
	} else {
		g.deepInevictable.Sub(cell.Size)
	}
	cell.SetDeepLoaded(false)
	
	// Remove from registry
	g.cellMutex.Lock()
	delete(g.cellRegistry, cell.ID)
	g.cellMutex.Unlock()
	
	log.Debug("cell evicted",
		zap.String("cellID", cell.ID),
		zap.Uint64("cellSize", cell.Size),
		zap.Bool("evictable", cell.Evictable),
		zap.Uint64("deepEvictable", g.deepEvictable.Load()),
		zap.Uint64("deepInevictable", g.deepInevictable.Load()),
	)
}

// OnCellLoadFailed should be called when a cell load fails
func (g *SegmentLoadGuard) OnCellLoadFailed(cell *Cell) {
	g.deepLoading.Sub(cell.Size)
	
	// Remove from registry
	g.cellMutex.Lock()
	delete(g.cellRegistry, cell.ID)
	g.cellMutex.Unlock()
	
	log.Debug("cell load failed",
		zap.String("cellID", cell.ID),
		zap.Uint64("cellSize", cell.Size),
		zap.Uint64("deepLoading", g.deepLoading.Load()),
	)
}

// OnSegmentReleased should be called when a segment is released
func (g *SegmentLoadGuard) OnSegmentReleased(segInfo *SegmentInfo) {
	if !segInfo.IsShallowLoaded.Load() {
		return
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	g.shallowInevictable -= segInfo.MetaSize + segInfo.InevictableSize
	g.shallowEvictable -= segInfo.EvictableSize
	segInfo.IsShallowLoaded.Store(false)

	// Also evict all cells belonging to this segment
	for _, cell := range segInfo.Cells {
		if cell.IsDeepLoaded() {
			g.OnCellEvicted(cell)
		}
	}

	log.Debug("segment released",
		zap.Int64("segmentID", segInfo.SegmentID),
		zap.Uint64("shallowEvictable", g.shallowEvictable),
		zap.Uint64("shallowInevictable", g.shallowInevictable),
	)
}

// tryEvict attempts to evict enough data to free the target amount of memory
func (g *SegmentLoadGuard) tryEvict(target int64) bool {
	if target <= 0 {
		return true
	}

	if !g.config.EvictionEnabled {
		return false
	}

	log.Debug("attempting to evict data",
		zap.Int64("targetBytes", target),
		zap.Uint64("currentEvictable", g.deepEvictable.Load()),
	)

	// Try to evict cells using a simple LRU-like strategy
	// In a real implementation, this would integrate with the caching layer
	// For now, we'll just check if we theoretically have enough evictable data
	if int64(g.deepEvictable.Load()) >= target {
		// TODO: Integrate with actual eviction mechanism
		// This is a placeholder - real implementation would trigger cache eviction
		return true
	}

	return false
}

// startEvictionLoop starts the background eviction loop
func (g *SegmentLoadGuard) startEvictionLoop() {
	g.evictionCtx, g.evictionCancel = context.WithCancel(context.Background())
	g.evictionDone = make(chan struct{})

	go func() {
		defer close(g.evictionDone)
		ticker := time.NewTicker(g.config.EvictionInterval)
		defer ticker.Stop()

		for {
			select {
			case <-g.evictionCtx.Done():
				return
			case <-ticker.C:
				g.evictionLoop()
			}
		}
	}()
}

// evictionLoop runs the periodic eviction check
func (g *SegmentLoadGuard) evictionLoop() {
	availableForCache := int64(g.limit) - int64(g.deepInevictable.Load())
	if availableForCache <= 0 {
		return
	}

	highWatermark := uint64(float64(availableForCache) * g.config.HighWatermark)
	if g.deepEvictable.Load() <= highWatermark {
		return
	}

	lowWatermark := uint64(float64(availableForCache) * g.config.LowWatermark)
	target := int64(g.deepEvictable.Load() - lowWatermark)

	log.Debug("eviction loop triggered",
		zap.Uint64("currentEvictable", g.deepEvictable.Load()),
		zap.Uint64("highWatermark", highWatermark),
		zap.Uint64("lowWatermark", lowWatermark),
		zap.Int64("targetEvict", target),
	)

	g.tryEvict(target)
}

// UpdateCellActualSize updates the actual size of a cell after it's loaded
func (g *SegmentLoadGuard) UpdateCellActualSize(cell *Cell, actualSize uint64) {
	if !cell.IsDeepLoaded() {
		return
	}

	sizeDiff := int64(actualSize) - int64(cell.Size)
	cell.Size = actualSize

	if cell.Evictable {
		if sizeDiff > 0 {
			g.deepEvictable.Add(uint64(sizeDiff))
		} else {
			g.deepEvictable.Sub(uint64(-sizeDiff))
		}
	} else {
		if sizeDiff > 0 {
			g.deepInevictable.Add(uint64(sizeDiff))
		} else {
			g.deepInevictable.Sub(uint64(-sizeDiff))
		}
	}

	log.Debug("cell size updated",
		zap.String("cellID", cell.ID),
		zap.Uint64("newSize", actualSize),
		zap.Int64("sizeDiff", sizeDiff),
	)
}

// GetStats returns current statistics
func (g *SegmentLoadGuard) GetStats() map[string]interface{} {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return map[string]interface{}{
		"limit":               g.limit,
		"shallow_evictable":   g.shallowEvictable,
		"shallow_inevictable": g.shallowInevictable,
		"deep_evictable":      g.deepEvictable.Load(),
		"deep_inevictable":    g.deepInevictable.Load(),
		"deep_loading":        g.deepLoading.Load(),
		"shallow_usage":       g.ShallowUsage(),
		"deep_usage":          g.DeepUsage(),
		"physical_usage":      g.memoryMonitor.GetUsedMemory(),
	}
}

// Close stops the load guard and cleans up resources
func (g *SegmentLoadGuard) Close() {
	if g.evictionCancel != nil {
		g.evictionCancel()
		<-g.evictionDone
	}
	g.memoryMonitor.Close()
}

// max returns the maximum of two int64 values
func max(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
} 