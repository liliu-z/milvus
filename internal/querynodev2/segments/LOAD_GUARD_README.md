# Caching Layer Load Guard Implementation

This document describes the implementation of the enhanced load guard system that moves resource protection from the Go segment loader to the C++ caching layer.

## Overview

The traditional Milvus resource protection system had limitations when the caching layer was introduced:

1. **Resource estimation inaccuracy**: Static resource estimation couldn't account for dynamic cache behavior
2. **Cache layer invisibility**: The Go-side loader couldn't see caching layer resource usage
3. **Pin mechanism complexity**: Resource pin/unpin lifecycle was difficult to manage
4. **Limited eviction coordination**: No coordination between loading and eviction policies

## New Architecture

### Core Components

1. **SegmentLoadGuard (C++)**: Core resource management logic
2. **CachingLayerLoadGuard (Go)**: Go wrapper with CGO bindings
3. **ResourceSynchronizer (Go)**: Cross-language resource state synchronization
4. **Enhanced Configuration**: New paramtable configuration parameters

### Key Concepts

#### Shallow vs Deep Loading
- **Shallow Loading**: Segment is marked as loaded but data remains in storage
- **Deep Loading**: Data is actually loaded into memory/disk on the node

#### Resource Types
- **Evictable**: Data that can be evicted when memory pressure occurs
- **Inevictable**: Data that must remain in memory (metadata, non-evictable indexes)

#### Water Levels
- **Logical Water Levels**: Based on estimated resource usage for admission control
- **Physical Water Levels**: Based on actual system resource usage for safety

## Implementation Details

### C++ Core (SegmentLoadGuard)

```cpp
class SegmentLoadGuard {
    // Shallow loading admission control
    bool CanLoadSegment(const Segment& segment);
    bool LoadSegment(std::shared_ptr<Segment> segment);
    
    // Deep loading admission control
    bool CanLoadCell(const Cell& cell);
    bool LoadCell(std::shared_ptr<Cell> cell);
    
    // Resource tracking
    ResourceUsage GetShallowUsage();
    ResourceUsage GetDeepUsage();
    ResourceUsage GetPhysicalUsage();
};
```

### Go Integration (load_guard_adapter.go)

```go
type CachingLayerLoadGuard struct {
    guard C.CSegmentLoadGuard
}

type LoadGuardConfig struct {
    OverloadPercentage   float64
    CacheRatio          float64
    EvictionEnabled     bool
    LowWatermark        float64
    HighWatermark       float64
    EvictionIntervalMs  int64
}
```

### Resource Synchronization (resource_sync.go)

```go
type ResourceSynchronizer struct {
    // Synchronizes resource state between Go and C++
    // Provides memory calibration using jemalloc stats
    // Handles cross-language resource tracking
}
```

## Configuration Parameters

### New Parameters in milvus.yaml

```yaml
queryNode:
  loadGuard:
    cacheRatio: 0.2              # Proportion of resources for cache
    lowWatermark: 0.7            # Eviction target threshold
    highWatermark: 0.8           # Eviction trigger threshold  
    evictionIntervalMs: 3000     # Eviction check interval
```

### Parameter Meanings

- **cacheRatio**: Controls the balance between resident and cached data
  - 0.0: No eviction benefit, all data must fit in memory
  - 1.0: Maximum eviction capacity, unlimited shallow loading
  
- **lowWatermark/highWatermark**: Control eviction aggressiveness
  - Higher values = more memory usage, less frequent eviction
  - Lower values = more conservative memory usage, frequent eviction

## Usage Examples

### Basic Usage

```go
// Create load guard with configuration
config := LoadGuardConfig{
    OverloadPercentage: 0.9,
    CacheRatio:        0.2,
    EvictionEnabled:   true,
    LowWatermark:     0.7,
    HighWatermark:    0.8,
    EvictionIntervalMs: 3000,
}

guard, err := NewCachingLayerLoadGuard(config)
if err != nil {
    return err
}
defer guard.Close()

// Set physical limits
err = guard.SetPhysicalLimits(8*GB, 100*GB)
if err != nil {
    return err
}
```

### Segment Loading

```go
// Check if segment can be loaded
segmentInfo := SegmentInfo{
    SegmentUID:      segmentID,
    MetaSize:        metaSize,
    EvictableSize:   evictableDataSize,
    InevictableSize: inevictableDataSize,
}

canLoad, err := guard.CanLoadSegment(segmentInfo)
if err != nil {
    return err
}

if canLoad {
    err = guard.LoadSegment(segmentInfo)
    if err != nil {
        return err
    }
}
```

### Resource Monitoring

```go
// Get resource usage report
shallowUsage, _ := guard.GetShallowUsage()
deepUsage, _ := guard.GetDeepUsage()
physicalUsage, _ := guard.GetPhysicalUsage()
available, _ := guard.GetAvailableResources()

fmt.Printf("Shallow: %dMB, Deep: %dMB, Physical: %dMB, Available: %dMB\n",
    shallowUsage.MemoryBytes/MB,
    deepUsage.MemoryBytes/MB, 
    physicalUsage.MemoryBytes/MB,
    available.MemoryBytes/MB)
```

## Deployment Scenarios

### High-Memory Systems
```yaml
queryNode:
  loadGuard:
    cacheRatio: 0.5        # Use more resources for caching
    lowWatermark: 0.6      # Aggressive eviction target
    highWatermark: 0.8     # Standard trigger
```

### Memory-Constrained Systems
```yaml
queryNode:
  loadGuard:
    cacheRatio: 0.1        # Conservative caching
    lowWatermark: 0.8      # Conservative eviction  
    highWatermark: 0.9     # Late trigger
```

### Disable Eviction (Traditional Behavior)
```yaml
queryNode:
  loadGuard:
    cacheRatio: 1.0        # All resources reserved
tieredStorage:
  evictionEnabled: false   # Disable eviction entirely
```

## Testing

### Unit Tests
```bash
cd internal/querynodev2/segments
go test -v -run TestLoadGuard*
```

### Integration Tests  
```bash
# Test with different configurations
cd internal/querynodev2/segments
go test -v -run TestConfigParameterIntegration
```

### C++ Tests
```bash
cd internal/core
make test-cachinglayer
```

## Performance Considerations

### Benefits
1. **Improved Resource Utilization**: Better coordination between loading and eviction
2. **Dynamic Resource Management**: Real-time adaptation to workload changes
3. **Cross-Language Synchronization**: Accurate resource tracking across Go/C++ boundary
4. **Configurable Policies**: Fine-tuned control over memory/performance trade-offs

### Overhead
1. **Cross-Language Calls**: CGO overhead for resource synchronization
2. **Background Threads**: Eviction and synchronization thread overhead
3. **Memory Tracking**: Additional memory usage for resource accounting

### Optimization Tips
1. **Tune Eviction Interval**: Balance responsiveness vs CPU overhead
2. **Adjust Cache Ratio**: Based on workload characteristics and available memory
3. **Monitor Resource Usage**: Use built-in monitoring for optimization guidance

## Migration Guide

### From Traditional Loader
1. **Update Configuration**: Add new load guard parameters
2. **Enable Tiered Storage**: Set `tieredStorage.evictionEnabled: true`
3. **Tune Parameters**: Start with defaults, adjust based on monitoring
4. **Monitor Behavior**: Watch resource usage patterns and adjust accordingly

### Backward Compatibility
- Default configuration maintains traditional behavior
- Existing parameters continue to work
- Gradual migration path available

## Troubleshooting

### Common Issues

1. **High Memory Usage**: 
   - Reduce `cacheRatio`
   - Lower `highWatermark`
   - Decrease `evictionIntervalMs`

2. **Frequent Evictions**:
   - Increase `cacheRatio`
   - Raise `lowWatermark`
   - Check if memory limits are appropriate

3. **Load Failures**:
   - Check physical limits configuration
   - Verify resource estimation accuracy
   - Monitor physical vs logical resource usage

### Debug Tools

```go
// Get comprehensive resource report
loader := // ... create synchronized loader
report, err := loader.GetResourceUsageReport()
if err == nil {
    for key, value := range report {
        fmt.Printf("%s: %v\n", key, value)
    }
}
```

### Monitoring Metrics
- `shallow_memory_mb`: Logically committed memory
- `deep_memory_mb`: Actually loaded memory  
- `physical_memory_mb`: System-reported memory usage
- `available_memory_mb`: Available for new loads
- `loaded_segment_count`: Number of loaded segments
- `loaded_cell_count`: Number of loaded cells

## Future Enhancements

1. **Advanced Eviction Policies**: LFU, workload-aware eviction
2. **Cross-Node Coordination**: Cluster-wide resource management
3. **Predictive Loading**: ML-based prefetching and eviction
4. **Storage Tiering**: Automatic data movement between storage tiers
5. **Resource Quotas**: Per-collection or per-user resource limits