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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadGuardConfig(t *testing.T) {
	t.Run("DefaultConfig", func(t *testing.T) {
		config := DefaultLoadGuardConfig()
		err := config.Validate()
		assert.NoError(t, err)
	})

	t.Run("InvalidConfig", func(t *testing.T) {
		// Test invalid cache ratio
		config := LoadGuardConfig{
			CacheRatio:         0,
			OverloadPercentage: 0.9,
			LowWatermark:       0.7,
			HighWatermark:      0.8,
		}
		err := config.Validate()
		assert.Error(t, err)

		// Test invalid watermarks
		config = LoadGuardConfig{
			CacheRatio:         0.5,
			OverloadPercentage: 0.9,
			LowWatermark:       0.8,
			HighWatermark:      0.7, // high < low
		}
		err = config.Validate()
		assert.Error(t, err)
	})
}

func TestCell(t *testing.T) {
	cell := NewCell("test_cell", 1024, true)
	
	assert.Equal(t, "test_cell", cell.ID)
	assert.Equal(t, uint64(1024), cell.Size)
	assert.True(t, cell.Evictable)
	assert.False(t, cell.IsDeepLoaded())
	
	cell.SetDeepLoaded(true)
	assert.True(t, cell.IsDeepLoaded())
}

func TestSegmentInfo(t *testing.T) {
	cells := []*Cell{
		NewCell("cell1", 1024, true),  // evictable
		NewCell("cell2", 2048, false), // not evictable
		NewCell("cell3", 512, true),   // evictable
	}
	
	segInfo := NewSegmentInfo(100, 256, cells)
	
	assert.Equal(t, int64(100), segInfo.SegmentID)
	assert.Equal(t, uint64(256), segInfo.MetaSize)
	assert.Equal(t, uint64(1024+512), segInfo.EvictableSize)
	assert.Equal(t, uint64(2048), segInfo.InevictableSize)
	assert.Equal(t, uint64(256+1024+512+2048), segInfo.TotalSize())
	assert.False(t, segInfo.IsShallowLoaded.Load())
}

func TestSegmentLoadGuard(t *testing.T) {
	config := LoadGuardConfig{
		OverloadPercentage: 0.8,
		CacheRatio:         0.3,
		EvictionEnabled:    false,
		LowWatermark:       0.6,
		HighWatermark:      0.7,
		EvictionInterval:   1 * time.Second,
	}
	
	guard, err := NewSegmentLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()
	
	t.Run("InitialState", func(t *testing.T) {
		assert.Equal(t, uint64(0), guard.ShallowUsage())
		assert.Equal(t, uint64(0), guard.DeepUsage())
		
		stats := guard.GetStats()
		assert.NotNil(t, stats)
		assert.Equal(t, uint64(0), stats["shallow_evictable"])
		assert.Equal(t, uint64(0), stats["shallow_inevictable"])
	})
	
	t.Run("SegmentAdmission", func(t *testing.T) {
		ctx := context.Background()
		
		// Create a small segment that should be admitted
		cells := []*Cell{
			NewCell("cell1", 1024, true),
		}
		segInfo := NewSegmentInfo(1, 256, cells)
		
		err := guard.SegmentLoadAdmission(ctx, segInfo)
		assert.NoError(t, err)
		assert.True(t, segInfo.IsShallowLoaded.Load())
		
		// Check that usage is updated
		expectedUsage := segInfo.MetaSize + segInfo.InevictableSize + 
			uint64(float64(segInfo.EvictableSize)*config.CacheRatio)
		assert.Equal(t, expectedUsage, guard.ShallowUsage())
		
		// Release the segment
		guard.OnSegmentReleased(segInfo)
		assert.False(t, segInfo.IsShallowLoaded.Load())
		assert.Equal(t, uint64(0), guard.ShallowUsage())
	})
	
	t.Run("CellAdmission", func(t *testing.T) {
		ctx := context.Background()
		
		cell := NewCell("test_cell", 512, true)
		
		err := guard.CellLoadAdmission(ctx, cell)
		assert.NoError(t, err)
		
		// Simulate successful load
		guard.OnCellLoaded(cell)
		assert.True(t, cell.IsDeepLoaded())
		assert.Equal(t, uint64(512), guard.DeepUsage())
		
		// Simulate eviction
		guard.OnCellEvicted(cell)
		assert.False(t, cell.IsDeepLoaded())
		assert.Equal(t, uint64(0), guard.DeepUsage())
	})
	
	t.Run("CellLoadFailure", func(t *testing.T) {
		ctx := context.Background()
		
		cell := NewCell("failing_cell", 512, true)
		
		err := guard.CellLoadAdmission(ctx, cell)
		assert.NoError(t, err)
		
		// Simulate load failure
		guard.OnCellLoadFailed(cell)
		assert.False(t, cell.IsDeepLoaded())
		assert.Equal(t, uint64(0), guard.DeepUsage())
	})
}

func TestSegmentLoadGuardWithEviction(t *testing.T) {
	config := LoadGuardConfig{
		OverloadPercentage: 0.8,
		CacheRatio:         0.5,
		EvictionEnabled:    true,
		LowWatermark:       0.3,
		HighWatermark:      0.6,
		EvictionInterval:   100 * time.Millisecond,
	}
	
	guard, err := NewSegmentLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()
	
	t.Run("EvictionLoop", func(t *testing.T) {
		ctx := context.Background()
		
		// Load several cells to trigger eviction
		for i := 0; i < 5; i++ {
			cell := NewCell(fmt.Sprintf("cell_%d", i), 1024, true)
			err := guard.CellLoadAdmission(ctx, cell)
			assert.NoError(t, err)
			guard.OnCellLoaded(cell)
		}
		
		// Wait for eviction loop to potentially run
		time.Sleep(200 * time.Millisecond)
		
		// Check that we have some loaded cells
		assert.Greater(t, guard.DeepUsage(), uint64(0))
	})
}

func TestMemoryMonitor(t *testing.T) {
	monitor := NewMemoryMonitorWithInterval(10 * time.Millisecond)
	defer monitor.Close()
	
	// Wait for at least one update
	time.Sleep(50 * time.Millisecond)
	
	usage := monitor.GetUsedMemory()
	assert.Greater(t, usage, uint64(0))
	
	// Test force update
	oldUsage := usage
	monitor.ForceUpdate()
	newUsage := monitor.GetUsedMemory()
	
	// Usage might be the same or different, but should be valid
	assert.Greater(t, newUsage, uint64(0))
}

func TestLoadGuardUtilFunctions(t *testing.T) {
	t.Run("ConvertResourceUsage", func(t *testing.T) {
		usage := ResourceUsage{
			MemorySize: 1024 * 1024, // 1MB
			DiskSize:   2048 * 1024, // 2MB
		}
		
		segInfo := ConvertResourceUsageToSegmentInfo(123, usage)
		assert.Equal(t, int64(123), segInfo.SegmentID)
		assert.Equal(t, uint64(1024*1024), segInfo.MetaSize) // 1MB metadata
		assert.Len(t, segInfo.Cells, 1)
		
		cell := segInfo.Cells[0]
		assert.Equal(t, "segment_123_legacy_data", cell.ID)
		assert.Equal(t, uint64(1024*1024), cell.Size)
		assert.True(t, cell.Evictable)
	})
} 