// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cachinglayer/Utils.h"
#include "common/EasyAssert.h"
#include "common/type_c.h"

namespace milvus::cachinglayer {

// Forward declarations
class Cell;
class Segment;

struct LoadGuardConfig {
    double overload_percentage = 0.9;      // 物理资源使用上限比例
    double cache_ratio = 0.2;              // 对于可驱逐数据，预留用于冷热交换的比例（不是总资源的比例）
    bool eviction_enabled = true;          // 是否启用驱逐
    double low_watermark = 0.7;           // 低水位线（基于可用于可驱逐数据的空间）
    double high_watermark = 0.8;          // 高水位线（基于可用于可驱逐数据的空间）
    std::chrono::milliseconds eviction_interval{3000};  // 驱逐检查间隔
    
    LoadGuardConfig() = default;
    LoadGuardConfig(double overload_pct, double cache_r, bool evict_enabled, 
                   double low_wm, double high_wm)
        : overload_percentage(overload_pct), cache_ratio(cache_r), 
          eviction_enabled(evict_enabled), low_watermark(low_wm), 
          high_watermark(high_wm) {}
};

// Enhanced Cell class with eviction support
class Cell {
public:
    Cell(int64_t size, bool evictable, uid_t uid = 0, cid_t cid = 0);
    ~Cell() = default;

    // Size management
    int64_t GetEstimatedSize() const { return estimated_size_; }
    int64_t GetActualSize() const { return actual_size_.load(); }
    void UpdateActualSize(int64_t actual_size);
    
    // Eviction properties
    bool IsEvictable() const { return evictable_; }
    bool IsDeepLoaded() const { return deep_loaded_.load(); }
    void SetDeepLoaded(bool loaded) { deep_loaded_.store(loaded); }
    
    // Identifiers
    uid_t GetUID() const { return uid_; }
    cid_t GetCID() const { return cid_; }

private:
    int64_t estimated_size_;
    std::atomic<int64_t> actual_size_;
    bool evictable_;
    std::atomic<bool> deep_loaded_{false};
    uid_t uid_;
    cid_t cid_;
};

// Enhanced Segment class for resource management
class Segment {
public:
    Segment(int64_t meta_size, std::vector<std::shared_ptr<Cell>> cells);
    ~Segment() = default;

    // Resource calculation
    int64_t GetMetaSize() const { return meta_size_; }
    int64_t GetEvictableSize() const;
    int64_t GetInevictableSize() const;
    int64_t GetTotalSize() const;
    
    // Cell management
    const std::vector<std::shared_ptr<Cell>>& GetCells() const { return cells_; }
    std::shared_ptr<Cell> GetCell(cid_t cid) const;
    
    // Load state
    bool IsLoaded() const { return loaded_.load(); }
    void SetLoaded(bool loaded) { loaded_.store(loaded); }

private:
    int64_t meta_size_;
    std::vector<std::shared_ptr<Cell>> cells_;
    std::atomic<bool> loaded_{false};
    
    // Cached size calculations (updated when cells change)
    mutable std::mutex size_cache_mutex_;
    mutable int64_t cached_evictable_size_{-1};
    mutable int64_t cached_inevictable_size_{-1};
};

// Main resource guard class
class SegmentLoadGuard {
public:
    explicit SegmentLoadGuard(const LoadGuardConfig& config);
    ~SegmentLoadGuard();

    // Configuration
    void UpdateConfig(const LoadGuardConfig& config);
    LoadGuardConfig GetConfig() const;
    
    // Physical resource limits
    void SetPhysicalLimits(int64_t memory_limit, int64_t disk_limit);
    std::pair<int64_t, int64_t> GetPhysicalLimits() const;
    
    // Segment load admission control
    bool CanLoadSegment(const Segment& segment) const;
    bool LoadSegment(std::shared_ptr<Segment> segment);
    void UnloadSegment(uid_t segment_uid);
    
    // Cell load admission control  
    bool CanLoadCell(const Cell& cell) const;
    bool LoadCell(std::shared_ptr<Cell> cell);
    void UnloadCell(uid_t segment_uid, cid_t cell_cid);
    
    // Resource usage queries
    ResourceUsage GetShallowUsage() const;
    ResourceUsage GetDeepUsage() const;
    ResourceUsage GetPhysicalUsage() const;
    ResourceUsage GetAvailableResources() const;
    
    // Eviction control
    bool TryEvict(int64_t target_memory, int64_t target_disk);
    void ForceEviction();
    
    // Status and monitoring
    size_t GetLoadedSegmentCount() const;
    size_t GetLoadedCellCount() const;
    std::vector<uid_t> GetLoadedSegmentUIDs() const;

private:
    // Configuration
    mutable std::shared_mutex config_mutex_;
    LoadGuardConfig config_;
    
    // Physical resource limits
    std::atomic<int64_t> memory_limit_{0};
    std::atomic<int64_t> disk_limit_{0};
    
    // Resource usage tracking
    std::atomic<ResourceUsage> shallow_evictable_{ResourceUsage(0, 0)};
    std::atomic<ResourceUsage> shallow_inevictable_{ResourceUsage(0, 0)};
    std::atomic<ResourceUsage> deep_evictable_{ResourceUsage(0, 0)};
    std::atomic<ResourceUsage> deep_inevictable_{ResourceUsage(0, 0)};
    std::atomic<ResourceUsage> deep_loading_{ResourceUsage(0, 0)};
    
    // Segment and cell tracking
    mutable std::shared_mutex segments_mutex_;
    std::unordered_map<uid_t, std::shared_ptr<Segment>> loaded_segments_;
    
    mutable std::shared_mutex cells_mutex_;
    std::unordered_map<uid_t, std::unordered_map<cid_t, std::shared_ptr<Cell>>> loaded_cells_;
    
    // Eviction thread
    std::atomic<bool> eviction_running_{false};
    std::unique_ptr<std::thread> eviction_thread_;
    
    // Helper methods
    ResourceUsage CalculateShallowUsage() const;
    ResourceUsage GetCurrentPhysicalUsage() const;
    bool CheckPhysicalLimits(const ResourceUsage& additional_usage) const;
    void StartEvictionThread();
    void StopEvictionThread();
    void EvictionLoop();
    bool ExecuteEviction(int64_t target_memory, int64_t target_disk);
    
    // Resource update helpers
    void UpdateShallowUsage(const Segment& segment, bool add);
    void UpdateDeepUsage(const Cell& cell, bool add, bool is_loading = false);
};

}  // namespace milvus::cachinglayer