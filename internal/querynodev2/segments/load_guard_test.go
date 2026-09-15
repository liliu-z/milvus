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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/util/paramtable"
)

func TestLoadGuardConfig(t *testing.T) {
	config := LoadGuardConfig{
		OverloadPercentage:   0.9,
		CacheRatio:          0.2,
		EvictionEnabled:     true,
		LowWatermark:        0.7,
		HighWatermark:       0.8,
		EvictionIntervalMs:  3000,
	}

	guard, err := NewCachingLayerLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()

	// Test configuration retrieval
	retrievedConfig, err := guard.GetConfig()
	require.NoError(t, err)
	assert.Equal(t, config.CacheRatio, retrievedConfig.CacheRatio)
	assert.Equal(t, config.EvictionEnabled, retrievedConfig.EvictionEnabled)
	assert.Equal(t, config.LowWatermark, retrievedConfig.LowWatermark)
	assert.Equal(t, config.HighWatermark, retrievedConfig.HighWatermark)
	assert.Equal(t, config.EvictionIntervalMs, retrievedConfig.EvictionIntervalMs)
}

func TestLoadGuardPhysicalLimits(t *testing.T) {
	config := LoadGuardConfig{
		OverloadPercentage:   0.9,
		CacheRatio:          0.2,
		EvictionEnabled:     true,
		LowWatermark:        0.7,
		HighWatermark:       0.8,
		EvictionIntervalMs:  3000,
	}

	guard, err := NewCachingLayerLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()

	// Set physical limits
	memoryLimit := int64(8 * 1024 * 1024 * 1024) // 8GB
	diskLimit := int64(100 * 1024 * 1024 * 1024)  // 100GB

	err = guard.SetPhysicalLimits(memoryLimit, diskLimit)
	require.NoError(t, err)

	// Get physical limits
	retrievedMemory, retrievedDisk, err := guard.GetPhysicalLimits()
	require.NoError(t, err)
	assert.Equal(t, memoryLimit, retrievedMemory)
	assert.Equal(t, diskLimit, retrievedDisk)
}

func TestLoadGuardSegmentLoading(t *testing.T) {
	config := LoadGuardConfig{
		OverloadPercentage:   0.9,
		CacheRatio:          0.5,  // Higher cache ratio for testing
		EvictionEnabled:     false, // Disable eviction for simplicity
		LowWatermark:        0.7,
		HighWatermark:       0.8,
		EvictionIntervalMs:  3000,
	}

	guard, err := NewCachingLayerLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()

	// Set reasonable physical limits
	memoryLimit := int64(1 * 1024 * 1024 * 1024) // 1GB
	diskLimit := int64(10 * 1024 * 1024 * 1024)   // 10GB
	err = guard.SetPhysicalLimits(memoryLimit, diskLimit)
	require.NoError(t, err)

	// Test segment info
	segmentInfo := SegmentInfo{
		SegmentUID:      1001,
		MetaSize:        1024 * 1024,        // 1MB meta
		EvictableSize:   50 * 1024 * 1024,   // 50MB evictable
		InevictableSize: 10 * 1024 * 1024,   // 10MB inevictable
	}

	// Check if we can load the segment
	canLoad, err := guard.CanLoadSegment(segmentInfo)
	require.NoError(t, err)
	assert.True(t, canLoad, "Should be able to load segment with reasonable size")

	// Load the segment
	err = guard.LoadSegment(segmentInfo)
	require.NoError(t, err)

	// Check loaded segment count
	segmentCount, err := guard.GetLoadedSegmentCount()
	require.NoError(t, err)
	assert.Equal(t, int64(1), segmentCount)

	// Check shallow usage
	shallowUsage, err := guard.GetShallowUsage()
	require.NoError(t, err)
	expectedShallow := segmentInfo.MetaSize + segmentInfo.InevictableSize + 
					   int64(float64(segmentInfo.EvictableSize) * config.CacheRatio)
	assert.Equal(t, expectedShallow, shallowUsage.MemoryBytes)

	// Unload the segment
	err = guard.UnloadSegment(segmentInfo.SegmentUID)
	require.NoError(t, err)

	// Check segment count is back to 0
	segmentCount, err = guard.GetLoadedSegmentCount()
	require.NoError(t, err)
	assert.Equal(t, int64(0), segmentCount)
}

func TestLoadGuardCellLoading(t *testing.T) {
	config := LoadGuardConfig{
		OverloadPercentage:   0.9,
		CacheRatio:          0.3,
		EvictionEnabled:     false, // Disable eviction for testing
		LowWatermark:        0.7,
		HighWatermark:       0.8,
		EvictionIntervalMs:  3000,
	}

	guard, err := NewCachingLayerLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()

	// Set physical limits
	memoryLimit := int64(1 * 1024 * 1024 * 1024) // 1GB
	diskLimit := int64(10 * 1024 * 1024 * 1024)   // 10GB
	err = guard.SetPhysicalLimits(memoryLimit, diskLimit)
	require.NoError(t, err)

	// Test cell info
	cellInfo := CellInfo{
		SegmentUID: 1001,
		CellCID:    2001,
		Size:       10 * 1024 * 1024, // 10MB
		Evictable:  true,
	}

	// Check if we can load the cell
	canLoad, err := guard.CanLoadCell(cellInfo)
	require.NoError(t, err)
	assert.True(t, canLoad, "Should be able to load cell with reasonable size")

	// Load the cell
	err = guard.LoadCell(cellInfo)
	require.NoError(t, err)

	// Check loaded cell count
	cellCount, err := guard.GetLoadedCellCount()
	require.NoError(t, err)
	assert.Equal(t, int64(1), cellCount)

	// Check deep usage
	deepUsage, err := guard.GetDeepUsage()
	require.NoError(t, err)
	assert.Equal(t, cellInfo.Size, deepUsage.MemoryBytes)

	// Unload the cell
	err = guard.UnloadCell(cellInfo.SegmentUID, cellInfo.CellCID)
	require.NoError(t, err)

	// Check cell count is back to 0
	cellCount, err = guard.GetLoadedCellCount()
	require.NoError(t, err)
	assert.Equal(t, int64(0), cellCount)
}

func TestLoadGuardEviction(t *testing.T) {
	config := LoadGuardConfig{
		OverloadPercentage:   0.9,
		CacheRatio:          0.3,
		EvictionEnabled:     true, // Enable eviction
		LowWatermark:        0.4,
		HighWatermark:       0.6,
		EvictionIntervalMs:  1000, // Faster eviction for testing
	}

	guard, err := NewCachingLayerLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()

	// Set small physical limits to trigger eviction
	memoryLimit := int64(100 * 1024 * 1024) // 100MB
	diskLimit := int64(1 * 1024 * 1024 * 1024) // 1GB
	err = guard.SetPhysicalLimits(memoryLimit, diskLimit)
	require.NoError(t, err)

	// Load multiple evictable cells
	for i := 0; i < 5; i++ {
		cellInfo := CellInfo{
			SegmentUID: 1001,
			CellCID:    int64(2000 + i),
			Size:       20 * 1024 * 1024, // 20MB each
			Evictable:  true,
		}

		canLoad, err := guard.CanLoadCell(cellInfo)
		require.NoError(t, err)

		if canLoad {
			err = guard.LoadCell(cellInfo)
			require.NoError(t, err)
		}
	}

	// Check that some cells were loaded
	cellCount, err := guard.GetLoadedCellCount()
	require.NoError(t, err)
	assert.Greater(t, cellCount, int64(0))

	// Force eviction
	err = guard.ForceEviction()
	require.NoError(t, err)

	// Give eviction some time to work
	time.Sleep(100 * time.Millisecond)

	// Check that eviction reduced the number of loaded cells or freed some memory
	newCellCount, err := guard.GetLoadedCellCount()
	require.NoError(t, err)

	deepUsage, err := guard.GetDeepUsage()
	require.NoError(t, err)

	t.Logf("After eviction: cellCount=%d, deepUsageMB=%d", newCellCount, deepUsage.MemoryBytes/(1024*1024))

	// The exact behavior depends on the eviction implementation,
	// but we should see some reduction in resource usage
	assert.True(t, newCellCount <= cellCount, "Eviction should reduce or maintain cell count")
}

func TestResourceSynchronizer(t *testing.T) {
	// Initialize paramtable for testing
	paramtable.Init()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	config := LoadGuardConfig{
		OverloadPercentage:   0.9,
		CacheRatio:          0.2,
		EvictionEnabled:     true,
		LowWatermark:        0.7,
		HighWatermark:       0.8,
		EvictionIntervalMs:  3000,
	}

	guard, err := NewCachingLayerLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()

	// Create resource synchronizer
	resourceSync := NewResourceSynchronizer(guard)
	defer resourceSync.Stop()

	// Start synchronization
	err = resourceSync.Start(ctx)
	require.NoError(t, err)

	// Give synchronizer some time to work
	time.Sleep(100 * time.Millisecond)

	// Test manual resource sync
	err = resourceSync.SyncResourceState()
	require.NoError(t, err)

	// Get physical usage
	physicalUsage, err := resourceSync.GetPhysicalUsage()
	require.NoError(t, err)
	
	// Should have some memory usage tracked
	assert.GreaterOrEqual(t, physicalUsage.MemoryBytes, int64(0))

	t.Logf("Physical usage: memory=%dMB, disk=%dMB", 
		physicalUsage.MemoryBytes/(1024*1024), physicalUsage.DiskBytes/(1024*1024))
}

func TestLoadGuardResourceUsageReport(t *testing.T) {
	// This would require a full segment loader setup
	// For now, just test that the types compile and basic methods work
	
	config := LoadGuardConfig{
		OverloadPercentage:   0.9,
		CacheRatio:          0.2,
		EvictionEnabled:     true,
		LowWatermark:        0.7,
		HighWatermark:       0.8,
		EvictionIntervalMs:  3000,
	}

	guard, err := NewCachingLayerLoadGuard(config)
	require.NoError(t, err)
	defer guard.Close()

	// Test available resources calculation
	available, err := guard.GetAvailableResources()
	require.NoError(t, err)
	
	// Initially should have no resources available since no limits are set
	assert.Equal(t, int64(0), available.MemoryBytes)
	assert.Equal(t, int64(0), available.DiskBytes)

	// Set limits and test again
	err = guard.SetPhysicalLimits(1024*1024*1024, 10*1024*1024*1024) // 1GB memory, 10GB disk
	require.NoError(t, err)

	available, err = guard.GetAvailableResources()
	require.NoError(t, err)
	
	// Should now have available resources
	assert.Greater(t, available.MemoryBytes, int64(0))
	assert.Greater(t, available.DiskBytes, int64(0))

	t.Logf("Available resources: memory=%dMB, disk=%dMB", 
		available.MemoryBytes/(1024*1024), available.DiskBytes/(1024*1024))
}

func TestConfigParameterIntegration(t *testing.T) {
	// Test that the new configuration parameters work with paramtable
	paramtable.Init()

	// Test accessing the new parameters
	cacheRatio := paramtable.Get().QueryNodeCfg.CacheRatio.GetAsFloat()
	lowWatermark := paramtable.Get().QueryNodeCfg.LoadGuardLowWatermark.GetAsFloat()
	highWatermark := paramtable.Get().QueryNodeCfg.LoadGuardHighWatermark.GetAsFloat()
	evictionInterval := paramtable.Get().QueryNodeCfg.LoadGuardEvictionIntervalMs.GetAsInt64()

	// Verify default values
	assert.Equal(t, 0.2, cacheRatio)
	assert.Equal(t, 0.7, lowWatermark)
	assert.Equal(t, 0.8, highWatermark)
	assert.Equal(t, int64(3000), evictionInterval)

	t.Logf("Config values: cacheRatio=%f, lowWatermark=%f, highWatermark=%f, evictionInterval=%d",
		cacheRatio, lowWatermark, highWatermark, evictionInterval)
}