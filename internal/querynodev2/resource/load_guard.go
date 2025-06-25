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

package resource

import (
	"context"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/hardware"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
)

// Cell represents a loadable unit within a segment.
// It can be a field's binlog, an index, or other data structures.
type Cell struct {
	// Estimated resource usage
	memSize int64
	diskSize int64

	// Actual resource usage after loaded
	actualMemSize int64
	actualDiskSize int64

	evictable   bool
	deepLoaded  bool
}

// Segment represents a segment to be loaded, containing multiple cells.
type Segment struct {
	id int64
	metaMemSize int64
	metaDiskSize int64
	cells []*Cell

	// Aggregated estimated sizes from cells
	evictableMemSize    int64
	inevictableMemSize  int64
	evictableDiskSize   int64
	inevictableDiskSize int64
}

// NewSegment creates a new segment for the load guard.
func NewSegment(id int64, metaMemSize, metaDiskSize int64, cells []*Cell) *Segment {
	s := &Segment{
		id: id,
		metaMemSize: metaMemSize,
		metaDiskSize: metaDiskSize,
		cells: cells,
	}

	for _, cell := range cells {
		if cell.evictable {
			s.evictableMemSize += cell.memSize
			s.evictableDiskSize += cell.diskSize
		} else {
			s.inevictableMemSize += cell.memSize
			s.inevictableDiskSize += cell.diskSize
		}
	}
	return s
}

// SegmentLoadGuard is a resource manager for segment loading on QueryNode.
// It provides fine-grained admission control for both segment "shallow load"
// and cell "deep load".
type SegmentLoadGuard struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	mut    sync.Mutex

	// Configuration
	cacheRatio           float64
	lowWaterMarkRatio    float64
	highWaterMarkRatio   float64
	evictionEnabled      bool

	// Resource limits
	memLimit  int64
	diskLimit int64

	// Logical usage: estimated resource usage of shallow-loaded segments
	shallowMemEvictable   int64
	shallowMemInevictable int64
	shallowDiskEvictable  int64
	shallowDiskInevictable int64

	// Physical usage: actual resource usage of deep-loaded cells
	deepMemEvictable   int64
	deepMemInevictable int64
	deepDiskEvictable  int64
	deepDiskInevictable int64

	// In-flight usage: estimated resource usage of cells being deep-loaded
	deepLoadingMem  int64
	deepLoadingDisk int64
}

// NewSegmentLoadGuard creates a new SegmentLoadGuard.
func NewSegmentLoadGuard(ctx context.Context, params *paramtable.QueryNodeConfig) (*SegmentLoadGuard, error) {
	ctx, cancel := context.WithCancel(ctx)

	guard := &SegmentLoadGuard{
		ctx:    ctx,
		cancel: cancel,

		evictionEnabled: params.Cache.GetEvictionEnabled(),
		cacheRatio: params.Cache.GetCacheRatio(),
		lowWaterMarkRatio: params.TieredStorage.GetMemoryLowWaterMark(),
		highWaterMarkRatio: params.TieredStorage.GetMemoryHighWaterMark(),

		memLimit:  int64(float64(hardware.GetMemoryCount()) * params.OverloadedMemoryThresholdPercentage.GetAsFloat()),
		diskLimit: int64(float64(params.DiskCapacityLimit.GetAsInt64()) * params.MaxDiskUsagePercentage.GetAsFloat()),
	}

	if !guard.evictionEnabled {
		guard.cacheRatio = 1.0
	}

	log.Info("SegmentLoadGuard created",
		zap.Bool("evictionEnabled", guard.evictionEnabled),
		zap.Float64("cacheRatio", guard.cacheRatio),
		zap.Int64("memLimit", guard.memLimit),
		zap.Int64("diskLimit", guard.diskLimit),
	)

	return guard, nil
}

// Start starts the background tasks for the guard.
func (g *SegmentLoadGuard) Start() {
	if g.evictionEnabled {
		g.wg.Add(1)
		go g.evictionLoop()
	}
}

// Stop stops the background tasks.
func (g *SegmentLoadGuard) Stop() {
	g.cancel()
	g.wg.Wait()
}


// SegmentLoad admits a segment for shallow-loading.
// It checks both logical and physical watermarks.
func (g *SegmentLoadGuard) SegmentLoad(seg *Segment) bool {
	g.mut.Lock()
	defer g.mut.Unlock()

	// 1. Calculate logical resource usage for the incoming segment.
	segLogicalMem := seg.metaMemSize + seg.inevictableMemSize + int64(float64(seg.evictableMemSize)*g.cacheRatio)
	segLogicalDisk := seg.metaDiskSize + seg.inevictableDiskSize + int64(float64(seg.evictableDiskSize)*g.cacheRatio)

	// 2. Check logical watermarks.
	if g.getLogicalMemUsage()+segLogicalMem > g.memLimit {
		log.Warn("Segment load rejected by logical memory limit", zap.Int64("segmentID", seg.id))
		return false
	}
	if g.getLogicalDiskUsage()+segLogicalDisk > g.diskLimit {
		log.Warn("Segment load rejected by logical disk limit", zap.Int64("segmentID", seg.id))
		return false
	}

	// 3. Check physical watermarks for loading metadata.
	if g.getCurrentPhysicalMemUsage()+seg.metaMemSize > g.memLimit {
		log.Warn("Segment load rejected by physical memory limit for metadata", zap.Int64("segmentID", seg.id))
		return false
	}
	if g.getCurrentPhysicalDiskUsage()+seg.metaDiskSize > g.diskLimit {
		log.Warn("Segment load rejected by physical disk limit for metadata", zap.Int64("segmentID", seg.id))
		return false
	}

	// 4. Admit success, update usages.
	g.shallowMemInevictable += seg.metaMemSize + seg.inevictableMemSize
	g.shallowMemEvictable += seg.evictableMemSize
	g.shallowDiskInevictable += seg.metaDiskSize + seg.inevictableDiskSize
	g.shallowDiskEvictable += seg.evictableDiskSize

	// Metadata is considered physically loaded immediately.
	g.deepMemInevictable += seg.metaMemSize
	g.deepDiskInevictable += seg.metaDiskSize

	return true
}

// OnSegmentReleaseOrLoadFail is called when a segment is released or its shallow-load fails.
func (g *SegmentLoadGuard) OnSegmentReleaseOrLoadFail(seg *Segment) {
	g.mut.Lock()
	defer g.mut.Unlock()

	g.shallowMemInevictable -= (seg.metaMemSize + seg.inevictableMemSize)
	g.shallowMemEvictable -= seg.evictableMemSize
	g.shallowDiskInevictable -= (seg.metaDiskSize + seg.inevictableDiskSize)
	g.shallowDiskEvictable -= seg.evictableDiskSize

	g.deepMemInevictable -= seg.metaMemSize
	g.deepDiskInevictable -= seg.metaDiskSize

	for _, cell := range seg.cells {
		if cell.deepLoaded {
			g.onCellEvictedOrReleased(cell)
		}
	}
}

// PermitLoadCell admits a cell for deep-loading.
func (g *SegmentLoadGuard) PermitLoadCell(cell *Cell) bool {
	g.mut.Lock()
	defer g.mut.Unlock()

	currentMemUsed := g.getCurrentPhysicalMemUsage() + g.deepLoadingMem
	currentDiskUsed := g.getCurrentPhysicalDiskUsage() + g.deepLoadingDisk

	availableMem := g.memLimit - currentMemUsed
	availableDisk := g.diskLimit - currentDiskUsed

	if cell.memSize <= availableMem && cell.diskSize <= availableDisk {
		g.deepLoadingMem += cell.memSize
		g.deepLoadingDisk += cell.diskSize
		return true
	}

	if !g.evictionEnabled || !cell.evictable {
		log.Warn("PermitLoadCell rejected, no eviction path", zap.Bool("evictionEnabled", g.evictionEnabled), zap.Bool("cellEvictable", cell.evictable))
		return false
	}
	
	// TODO: implement tryEvict
	return false
}

// OnCellLoaded is called after a cell is successfully deep-loaded.
func (g *SegmentLoadGuard) OnCellLoaded(cell *Cell, actualMemSize, actualDiskSize int64) {
	g.mut.Lock()
	defer g.mut.Unlock()
	
	cell.actualMemSize = actualMemSize
	cell.actualDiskSize = actualDiskSize
	cell.deepLoaded = true

	// Move from loading to loaded state.
	g.deepLoadingMem -= cell.memSize
	g.deepLoadingDisk -= cell.diskSize

	if cell.evictable {
		g.deepMemEvictable += actualMemSize
		g.deepDiskEvictable += actualDiskSize
	} else {
		g.deepMemInevictable += actualMemSize
		g.deepDiskInevictable += actualDiskSize
	}

	// TODO: check for overflow and trigger emergency eviction.
}

// onCellEvictedOrReleased is an internal method called when a cell is evicted or its segment is released.
func (g *SegmentLoadGuard) onCellEvictedOrReleased(cell *Cell) {
	if !cell.deepLoaded {
		return
	}

	memToRelease := cell.actualMemSize
	if memToRelease == 0 {
		memToRelease = cell.memSize
	}
	diskToRelease := cell.actualDiskSize
	if diskToRelease == 0 {
		diskToRelease = cell.diskSize
	}

	if cell.evictable {
		g.deepMemEvictable -= memToRelease
		g.deepDiskEvictable -= diskToRelease
	} else {
		g.deepMemInevictable -= memToRelease
		g.deepDiskInevictable -= diskToRelease
	}

	cell.deepLoaded = false
	cell.actualMemSize = 0
	cell.actualDiskSize = 0
}

// OnCellLoadFail is called when a cell's deep-load fails.
func (g *SegmentLoadGuard) OnCellLoadFail(cell *Cell) {
	g.mut.Lock()
	defer g.mut.Unlock()

	g.deepLoadingMem -= cell.memSize
	g.deepLoadingDisk -= cell.diskSize
}


// evictionLoop is the background routine for periodic eviction.
func (g *SegmentLoadGuard) evictionLoop() {
	defer g.wg.Done()
	ticker := time.NewTicker(3 * time.Second) // TODO: make configurable
	defer ticker.Stop()

	log.Info("SegmentLoadGuard eviction loop started")

	for {
		select {
		case <-g.ctx.Done():
			log.Info("SegmentLoadGuard eviction loop stopped")
			return
		case <-ticker.C:
			// TODO: implement eviction logic
		}
	}
}

// --- Helper functions ---

func (g *SegmentLoadGuard) getLogicalMemUsage() int64 {
	return g.shallowMemInevictable + int64(float64(g.shallowMemEvictable)*g.cacheRatio)
}

func (g *SegmentLoadGuard) getLogicalDiskUsage() int64 {
	return g.shallowDiskInevictable + int64(float64(g.shallowDiskEvictable)*g.cacheRatio)
}

func (g *SegmentLoadGuard) getCurrentPhysicalMemUsage() int64 {
	// For memory, OS usage is a fallback.
	used, _ := hardware.GetUsedMemoryCount()
	return max(int64(used), g.deepMemEvictable+g.deepMemInevictable)
}

func (g *SegmentLoadGuard) getCurrentPhysicalDiskUsage() int64 {
	// For disk, we assume our tracking is accurate.
	return g.deepDiskEvictable + g.deepDiskInevictable
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
} 