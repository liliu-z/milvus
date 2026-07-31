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

#include "cachinglayer/SegmentLoadGuard.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#include "log/Log.h"

namespace milvus::cachinglayer {

//=============================================================================
// Cell Implementation
//=============================================================================

Cell::Cell(int64_t size, bool evictable, uid_t uid, cid_t cid)
    : estimated_size_(size), actual_size_(size), evictable_(evictable), 
      uid_(uid), cid_(cid) {
    AssertInfo(size >= 0, "Cell size must be non-negative");
}

void Cell::UpdateActualSize(int64_t actual_size) {
    AssertInfo(actual_size >= 0, "Cell actual size must be non-negative");
    actual_size_.store(actual_size);
}

//=============================================================================
// Segment Implementation  
//=============================================================================

Segment::Segment(int64_t meta_size, std::vector<std::shared_ptr<Cell>> cells)
    : meta_size_(meta_size), cells_(std::move(cells)) {
    AssertInfo(meta_size >= 0, "Segment meta size must be non-negative");
}

int64_t Segment::GetEvictableSize() const {
    std::lock_guard<std::mutex> lock(size_cache_mutex_);
    if (cached_evictable_size_ < 0) {
        cached_evictable_size_ = 0;
        for (const auto& cell : cells_) {
            if (cell->IsEvictable()) {
                cached_evictable_size_ += cell->GetActualSize();
            }
        }
    }
    return cached_evictable_size_;
}

int64_t Segment::GetInevictableSize() const {
    std::lock_guard<std::mutex> lock(size_cache_mutex_);
    if (cached_inevictable_size_ < 0) {
        cached_inevictable_size_ = meta_size_;
        for (const auto& cell : cells_) {
            if (!cell->IsEvictable()) {
                cached_inevictable_size_ += cell->GetActualSize();
            }
        }
    }
    return cached_inevictable_size_;
}

int64_t Segment::GetTotalSize() const {
    return GetEvictableSize() + GetInevictableSize();
}

std::shared_ptr<Cell> Segment::GetCell(cid_t cid) const {
    for (const auto& cell : cells_) {
        if (cell->GetCID() == cid) {
            return cell;
        }
    }
    return nullptr;
}

//=============================================================================
// SegmentLoadGuard Implementation
//=============================================================================

SegmentLoadGuard::SegmentLoadGuard(const LoadGuardConfig& config) 
    : config_(config) {
    // Validate configuration
    AssertInfo(config_.overload_percentage > 0 && config_.overload_percentage <= 1.0,
               "overload_percentage must be in (0, 1], got " + std::to_string(config_.overload_percentage));
    AssertInfo(config_.cache_ratio >= 0 && config_.cache_ratio <= 1.0,
               "cache_ratio must be in [0, 1], got " + std::to_string(config_.cache_ratio));
    AssertInfo(config_.low_watermark >= 0 && config_.low_watermark <= 1.0,
               "low_watermark must be in [0, 1], got " + std::to_string(config_.low_watermark));
    AssertInfo(config_.high_watermark >= 0 && config_.high_watermark <= 1.0,
               "high_watermark must be in [0, 1], got " + std::to_string(config_.high_watermark));
    AssertInfo(config_.low_watermark <= config_.high_watermark,
               "low_watermark must be <= high_watermark, got low=" + std::to_string(config_.low_watermark) + 
               ", high=" + std::to_string(config_.high_watermark));
    
    LOG_INFO("SegmentLoadGuard initialized with cache_ratio={}, eviction_enabled={}", 
             config_.cache_ratio, config_.eviction_enabled);
    
    if (config_.eviction_enabled) {
        StartEvictionThread();
    }
}

SegmentLoadGuard::~SegmentLoadGuard() {
    StopEvictionThread();
}

void SegmentLoadGuard::UpdateConfig(const LoadGuardConfig& config) {
    // Validate configuration before updating
    AssertInfo(config.overload_percentage > 0 && config.overload_percentage <= 1.0,
               "overload_percentage must be in (0, 1], got " + std::to_string(config.overload_percentage));
    AssertInfo(config.cache_ratio >= 0 && config.cache_ratio <= 1.0,
               "cache_ratio must be in [0, 1], got " + std::to_string(config.cache_ratio));
    AssertInfo(config.low_watermark >= 0 && config.low_watermark <= 1.0,
               "low_watermark must be in [0, 1], got " + std::to_string(config.low_watermark));
    AssertInfo(config.high_watermark >= 0 && config.high_watermark <= 1.0,
               "high_watermark must be in [0, 1], got " + std::to_string(config.high_watermark));
    AssertInfo(config.low_watermark <= config.high_watermark,
               "low_watermark must be <= high_watermark, got low=" + std::to_string(config.low_watermark) + 
               ", high=" + std::to_string(config.high_watermark));
    
    {
        std::unique_lock<std::shared_mutex> lock(config_mutex_);
        bool eviction_state_changed = (config_.eviction_enabled != config.eviction_enabled);
        config_ = config;
        
        if (eviction_state_changed) {
            if (config_.eviction_enabled) {
                StartEvictionThread();
            } else {
                StopEvictionThread();
            }
        }
    }
    LOG_INFO("SegmentLoadGuard config updated: cache_ratio={}, eviction_enabled={}", 
             config_.cache_ratio, config_.eviction_enabled);
}

LoadGuardConfig SegmentLoadGuard::GetConfig() const {
    std::shared_lock<std::shared_mutex> lock(config_mutex_);
    return config_;
}

void SegmentLoadGuard::SetPhysicalLimits(int64_t memory_limit, int64_t disk_limit) {
    memory_limit_.store(memory_limit);
    disk_limit_.store(disk_limit);
    LOG_INFO("Physical limits updated: memory={}GB, disk={}GB", 
             memory_limit / (1024.0 * 1024.0 * 1024.0),
             disk_limit / (1024.0 * 1024.0 * 1024.0));
}

std::pair<int64_t, int64_t> SegmentLoadGuard::GetPhysicalLimits() const {
    return {memory_limit_.load(), disk_limit_.load()};
}

bool SegmentLoadGuard::CanLoadSegment(const Segment& segment) const {
    auto config = GetConfig();
    
    // Calculate current shallow usage with cache ratio applied
    ResourceUsage current_shallow_usage;
    current_shallow_usage.memory_bytes = shallow_inevictable_.load().memory_bytes + 
                                        static_cast<int64_t>(shallow_evictable_.load().memory_bytes * config.cache_ratio);
    current_shallow_usage.file_bytes = shallow_inevictable_.load().file_bytes + 
                                      static_cast<int64_t>(shallow_evictable_.load().file_bytes * config.cache_ratio);
    
    // Calculate segment's shallow usage
    ResourceUsage segment_shallow_usage;
    segment_shallow_usage.memory_bytes = segment.GetInevictableSize() + 
                                       static_cast<int64_t>(segment.GetEvictableSize() * config.cache_ratio);
    segment_shallow_usage.file_bytes = 0; // Disk usage calculated separately if needed
    
    auto total_shallow = current_shallow_usage + segment_shallow_usage;
    
    // Check against logical limits
    auto [memory_limit, disk_limit] = GetPhysicalLimits();
    int64_t logical_memory_limit = static_cast<int64_t>(memory_limit * config.overload_percentage);
    int64_t logical_disk_limit = static_cast<int64_t>(disk_limit * config.overload_percentage);
    
    if (total_shallow.memory_bytes > logical_memory_limit || 
        total_shallow.file_bytes > logical_disk_limit) {
        return false;
    }
    
    // Check physical limits as additional safety
    auto physical_usage = GetCurrentPhysicalUsage();
    return CheckPhysicalLimits(segment_shallow_usage);
}

bool SegmentLoadGuard::LoadSegment(std::shared_ptr<Segment> segment) {
    if (!CanLoadSegment(*segment)) {
        LOG_WARN("Cannot load segment {}: resource limits exceeded", segment->GetUID());
        return false;
    }
    
    {
        std::unique_lock<std::shared_mutex> lock(segments_mutex_);
        loaded_segments_[segment->GetUID()] = segment;
    }
    
    UpdateShallowUsage(*segment, true);
    segment->SetLoaded(true);
    
    LOG_DEBUG("Segment {} loaded successfully", segment->GetUID());
    return true;
}

void SegmentLoadGuard::UnloadSegment(uid_t segment_uid) {
    std::shared_ptr<Segment> segment;
    
    {
        std::unique_lock<std::shared_mutex> lock(segments_mutex_);
        auto it = loaded_segments_.find(segment_uid);
        if (it == loaded_segments_.end()) {
            LOG_WARN("Segment {} not found for unloading", segment_uid);
            return;
        }
        segment = it->second;
        loaded_segments_.erase(it);
    }
    
    // Unload all cells in this segment
    {
        std::unique_lock<std::shared_mutex> cell_lock(cells_mutex_);
        auto cell_it = loaded_cells_.find(segment_uid);
        if (cell_it != loaded_cells_.end()) {
            for (const auto& [cid, cell] : cell_it->second) {
                if (cell->IsDeepLoaded()) {
                    UpdateDeepUsage(*cell, false);
                    cell->SetDeepLoaded(false);
                }
            }
            loaded_cells_.erase(cell_it);
        }
    }
    
    UpdateShallowUsage(*segment, false);
    segment->SetLoaded(false);
    
    LOG_DEBUG("Segment {} unloaded successfully", segment_uid);
}

bool SegmentLoadGuard::CanLoadCell(const Cell& cell) const {
    // Physical resource check with deep loading consideration
    auto current_deep = deep_evictable_.load() + deep_inevictable_.load();
    auto current_loading = deep_loading_.load();
    auto physical_usage = GetCurrentPhysicalUsage();
    
    ResourceUsage required_usage;
    required_usage.memory_bytes = cell.GetActualSize();
    required_usage.file_bytes = 0; // Adjust based on cell type if needed
    
    auto [memory_limit, disk_limit] = GetPhysicalLimits();
    auto available_memory = memory_limit - physical_usage.memory_bytes - current_loading.memory_bytes;
    auto available_disk = disk_limit - physical_usage.file_bytes - current_loading.file_bytes;
    
    return required_usage.memory_bytes <= available_memory && 
           required_usage.file_bytes <= available_disk;
}

bool SegmentLoadGuard::LoadCell(std::shared_ptr<Cell> cell) {
    if (!CanLoadCell(*cell)) {
        // Try eviction if needed and possible
        auto config = GetConfig();
        if (config.eviction_enabled) {
            ResourceUsage required;
            required.memory_bytes = cell->GetActualSize();
            required.file_bytes = 0;
            
            auto available = GetAvailableResources();
            if (required.memory_bytes > available.memory_bytes) {
                int64_t need_to_evict = required.memory_bytes - available.memory_bytes;
                if (!TryEvict(need_to_evict, 0)) {
                    LOG_WARN("Cannot load cell {}: eviction failed", cell->GetCID());
                    return false;
                }
            }
        } else {
            LOG_WARN("Cannot load cell {}: resource limits exceeded", cell->GetCID());
            return false;
        }
    }
    
    // Mark as loading
    UpdateDeepUsage(*cell, true, true);
    
    // Simulate loading process here
    // In real implementation, this would trigger actual data loading
    
    // Mark as loaded: first remove from loading, then add to loaded
    UpdateDeepUsage(*cell, false, true);  // Remove from loading
    UpdateDeepUsage(*cell, true, false);  // Add to loaded
    cell->SetDeepLoaded(true);
    
    {
        std::unique_lock<std::shared_mutex> lock(cells_mutex_);
        loaded_cells_[cell->GetUID()][cell->GetCID()] = cell;
    }
    
    LOG_DEBUG("Cell {} loaded successfully", cell->GetCID());
    return true;
}

void SegmentLoadGuard::UnloadCell(uid_t segment_uid, cid_t cell_cid) {
    std::shared_ptr<Cell> cell;
    
    {
        std::unique_lock<std::shared_mutex> lock(cells_mutex_);
        auto seg_it = loaded_cells_.find(segment_uid);
        if (seg_it == loaded_cells_.end()) {
            LOG_WARN("Segment {} not found for cell unloading", segment_uid);
            return;
        }
        
        auto cell_it = seg_it->second.find(cell_cid);
        if (cell_it == seg_it->second.end()) {
            LOG_WARN("Cell {} not found in segment {} for unloading", cell_cid, segment_uid);
            return;
        }
        
        cell = cell_it->second;
        seg_it->second.erase(cell_it);
        
        if (seg_it->second.empty()) {
            loaded_cells_.erase(seg_it);
        }
    }
    
    if (cell->IsDeepLoaded()) {
        UpdateDeepUsage(*cell, false);
        cell->SetDeepLoaded(false);
    }
    
    LOG_DEBUG("Cell {} in segment {} unloaded successfully", cell_cid, segment_uid);
}

ResourceUsage SegmentLoadGuard::GetShallowUsage() const {
    auto config = GetConfig();
    ResourceUsage usage;
    usage.memory_bytes = shallow_inevictable_.load().memory_bytes + 
                        static_cast<int64_t>(shallow_evictable_.load().memory_bytes * config.cache_ratio);
    usage.file_bytes = shallow_inevictable_.load().file_bytes + 
                      static_cast<int64_t>(shallow_evictable_.load().file_bytes * config.cache_ratio);
    return usage;
}

ResourceUsage SegmentLoadGuard::GetDeepUsage() const {
    return deep_evictable_.load() + deep_inevictable_.load();
}

ResourceUsage SegmentLoadGuard::GetPhysicalUsage() const {
    return GetCurrentPhysicalUsage();
}

ResourceUsage SegmentLoadGuard::GetAvailableResources() const {
    auto [memory_limit, disk_limit] = GetPhysicalLimits();
    auto physical_usage = GetCurrentPhysicalUsage();
    auto loading_usage = deep_loading_.load();
    
    ResourceUsage available;
    available.memory_bytes = std::max(0L, memory_limit - physical_usage.memory_bytes - loading_usage.memory_bytes);
    available.file_bytes = std::max(0L, disk_limit - physical_usage.file_bytes - loading_usage.file_bytes);
    
    return available;
}

bool SegmentLoadGuard::TryEvict(int64_t target_memory, int64_t target_disk) {
    return ExecuteEviction(target_memory, target_disk);
}

void SegmentLoadGuard::ForceEviction() {
    auto config = GetConfig();
    auto deep_evictable = deep_evictable_.load();
    auto deep_inevictable = deep_inevictable_.load();
    auto [memory_limit, disk_limit] = GetPhysicalLimits();
    
    // Calculate available space for evictable data
    int64_t available_memory = memory_limit - deep_inevictable.memory_bytes;
    int64_t available_disk = disk_limit - deep_inevictable.file_bytes;
    
    int64_t target_memory = deep_evictable.memory_bytes - static_cast<int64_t>(available_memory * config.low_watermark);
    int64_t target_disk = deep_evictable.file_bytes - static_cast<int64_t>(available_disk * config.low_watermark);
    
    if (target_memory > 0 || target_disk > 0) {
        ExecuteEviction(std::max(0L, target_memory), std::max(0L, target_disk));
    }
}

size_t SegmentLoadGuard::GetLoadedSegmentCount() const {
    std::shared_lock<std::shared_mutex> lock(segments_mutex_);
    return loaded_segments_.size();
}

size_t SegmentLoadGuard::GetLoadedCellCount() const {
    std::shared_lock<std::shared_mutex> lock(cells_mutex_);
    size_t count = 0;
    for (const auto& [seg_uid, cells] : loaded_cells_) {
        count += cells.size();
    }
    return count;
}

std::vector<uid_t> SegmentLoadGuard::GetLoadedSegmentUIDs() const {
    std::shared_lock<std::shared_mutex> lock(segments_mutex_);
    std::vector<uid_t> uids;
    uids.reserve(loaded_segments_.size());
    for (const auto& [uid, segment] : loaded_segments_) {
        uids.push_back(uid);
    }
    return uids;
}

//=============================================================================
// Private Helper Methods
//=============================================================================

ResourceUsage SegmentLoadGuard::CalculateShallowUsage() const {
    return shallow_evictable_.load() + shallow_inevictable_.load();
}

ResourceUsage SegmentLoadGuard::GetCurrentPhysicalUsage() const {
    ResourceUsage usage;
    
    // Get memory usage from system
#ifdef __GLIBC__
    struct mallinfo2 mi = mallinfo2();
    usage.memory_bytes = mi.hblkhd + mi.uordblks;
#else
    // Fallback to a simple estimation
    usage.memory_bytes = (deep_evictable_.load() + deep_inevictable_.load()).memory_bytes;
#endif
    
    // Disk usage would need to be implemented based on actual file system usage
    usage.file_bytes = (deep_evictable_.load() + deep_inevictable_.load()).file_bytes;
    
    return usage;
}

bool SegmentLoadGuard::CheckPhysicalLimits(const ResourceUsage& additional_usage) const {
    auto current = GetCurrentPhysicalUsage();
    auto total = current + additional_usage;
    auto [memory_limit, disk_limit] = GetPhysicalLimits();
    
    return total.memory_bytes <= memory_limit && total.file_bytes <= disk_limit;
}

void SegmentLoadGuard::StartEvictionThread() {
    if (eviction_running_.load()) {
        return;
    }
    
    eviction_running_.store(true);
    eviction_thread_ = std::make_unique<std::thread>(&SegmentLoadGuard::EvictionLoop, this);
    LOG_INFO("Eviction thread started");
}

void SegmentLoadGuard::StopEvictionThread() {
    if (!eviction_running_.load()) {
        return;
    }
    
    eviction_running_.store(false);
    if (eviction_thread_ && eviction_thread_->joinable()) {
        eviction_thread_->join();
    }
    eviction_thread_.reset();
    LOG_INFO("Eviction thread stopped");
}

void SegmentLoadGuard::EvictionLoop() {
    auto config = GetConfig();
    
    while (eviction_running_.load()) {
        try {
            auto deep_evictable = deep_evictable_.load();
            auto deep_inevictable = deep_inevictable_.load();
            auto [memory_limit, disk_limit] = GetPhysicalLimits();
            
            // Calculate available space for evictable data
            int64_t available_memory = memory_limit - deep_inevictable.memory_bytes;
            int64_t available_disk = disk_limit - deep_inevictable.file_bytes;
            
            // Check if we need eviction based on evictable data only
            bool need_eviction = false;
            int64_t target_memory = 0, target_disk = 0;
            
            if (deep_evictable.memory_bytes > available_memory * config.high_watermark) {
                need_eviction = true;
                target_memory = deep_evictable.memory_bytes - static_cast<int64_t>(available_memory * config.low_watermark);
            }
            
            if (deep_evictable.file_bytes > available_disk * config.high_watermark) {
                need_eviction = true;
                target_disk = deep_evictable.file_bytes - static_cast<int64_t>(available_disk * config.low_watermark);
            }
            
            if (need_eviction) {
                LOG_DEBUG("Starting eviction: target_memory={}MB, target_disk={}MB", 
                         target_memory / (1024 * 1024), target_disk / (1024 * 1024));
                ExecuteEviction(target_memory, target_disk);
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("Error in eviction loop: {}", e.what());
        }
        
        std::this_thread::sleep_for(config.eviction_interval);
    }
}

bool SegmentLoadGuard::ExecuteEviction(int64_t target_memory, int64_t target_disk) {
    // This is a simplified eviction implementation
    // In practice, this would integrate with the existing LRU eviction mechanism
    
    int64_t evicted_memory = 0, evicted_disk = 0;
    std::vector<std::pair<uid_t, cid_t>> cells_to_evict;
    
    {
        std::shared_lock<std::shared_mutex> lock(cells_mutex_);
        for (const auto& [seg_uid, cells] : loaded_cells_) {
            for (const auto& [cell_cid, cell] : cells) {
                if (cell->IsEvictable() && cell->IsDeepLoaded()) {
                    cells_to_evict.emplace_back(seg_uid, cell_cid);
                    evicted_memory += cell->GetActualSize();
                    
                    if (evicted_memory >= target_memory && evicted_disk >= target_disk) {
                        break;
                    }
                }
            }
            if (evicted_memory >= target_memory && evicted_disk >= target_disk) {
                break;
            }
        }
    }
    
    // Actually evict the cells
    for (const auto& [seg_uid, cell_cid] : cells_to_evict) {
        UnloadCell(seg_uid, cell_cid);
        
        if (evicted_memory >= target_memory && evicted_disk >= target_disk) {
            break;
        }
    }
    
    bool success = (evicted_memory >= target_memory) && (evicted_disk >= target_disk);
    LOG_DEBUG("Eviction completed: evicted_memory={}MB, evicted_disk={}MB, success={}", 
             evicted_memory / (1024 * 1024), evicted_disk / (1024 * 1024), success);
    
    return success;
}

void SegmentLoadGuard::UpdateShallowUsage(const Segment& segment, bool add) {
    ResourceUsage evictable_delta, inevictable_delta;
    
    evictable_delta.memory_bytes = segment.GetEvictableSize();
    inevictable_delta.memory_bytes = segment.GetInevictableSize();
    
    if (!add) {
        evictable_delta.memory_bytes = -evictable_delta.memory_bytes;
        inevictable_delta.memory_bytes = -inevictable_delta.memory_bytes;
    }
    
    shallow_evictable_ += evictable_delta;
    shallow_inevictable_ += inevictable_delta;
}

void SegmentLoadGuard::UpdateDeepUsage(const Cell& cell, bool add, bool is_loading) {
    ResourceUsage delta;
    delta.memory_bytes = cell.GetActualSize();
    delta.file_bytes = 0; // Adjust based on cell storage type if needed
    
    if (!add) {
        delta.memory_bytes = -delta.memory_bytes;
        delta.file_bytes = -delta.file_bytes;
    }
    
    if (is_loading) {
        deep_loading_ += delta;
    } else {
        if (cell.IsEvictable()) {
            deep_evictable_ += delta;
        } else {
            deep_inevictable_ += delta;
        }
    }
}

}  // namespace milvus::cachinglayer