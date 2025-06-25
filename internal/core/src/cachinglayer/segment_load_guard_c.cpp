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

#include "cachinglayer/segment_load_guard_c.h"
#include "cachinglayer/SegmentLoadGuard.h"

#include <thread>
#include <exception>
#include <memory>

using namespace milvus::cachinglayer;

namespace {
    thread_local std::string last_error_;
    
    void SetLastError(const std::string& error) {
        last_error_ = error;
    }
    
    template<typename Func>
    auto SafeExecute(Func&& func) -> decltype(func()) {
        try {
            return func();
        } catch (const std::exception& e) {
            SetLastError(e.what());
            if constexpr (std::is_same_v<decltype(func()), bool>) {
                return false;
            } else if constexpr (std::is_same_v<decltype(func()), void>) {
                return;
            } else {
                return {};
            }
        }
    }
    
    LoadGuardConfig ConvertConfig(const CLoadGuardConfig& c_config) {
        LoadGuardConfig config;
        config.overload_percentage = c_config.overload_percentage;
        config.cache_ratio = c_config.cache_ratio;
        config.eviction_enabled = c_config.eviction_enabled;
        config.low_watermark = c_config.low_watermark;
        config.high_watermark = c_config.high_watermark;
        config.eviction_interval = std::chrono::milliseconds(c_config.eviction_interval_ms);
        return config;
    }
    
    CLoadGuardConfig ConvertConfig(const LoadGuardConfig& config) {
        CLoadGuardConfig c_config;
        c_config.overload_percentage = config.overload_percentage;
        c_config.cache_ratio = config.cache_ratio;
        c_config.eviction_enabled = config.eviction_enabled;
        c_config.low_watermark = config.low_watermark;
        c_config.high_watermark = config.high_watermark;
        c_config.eviction_interval_ms = config.eviction_interval.count();
        return c_config;
    }
    
    CResourceUsage ConvertResourceUsage(const ResourceUsage& usage) {
        CResourceUsage c_usage;
        c_usage.memory_bytes = usage.memory_bytes;
        c_usage.disk_bytes = usage.file_bytes;
        return c_usage;
    }
}

extern "C" {

CSegmentLoadGuard NewSegmentLoadGuard(CLoadGuardConfig config) {
    return SafeExecute([&]() -> CSegmentLoadGuard {
        auto cpp_config = ConvertConfig(config);
        auto guard = std::make_unique<SegmentLoadGuard>(cpp_config);
        return guard.release();
    });
}

void DeleteSegmentLoadGuard(CSegmentLoadGuard guard) {
    SafeExecute([&]() {
        if (guard) {
            delete static_cast<SegmentLoadGuard*>(guard);
        }
    });
}

void UpdateLoadGuardConfig(CSegmentLoadGuard guard, CLoadGuardConfig config) {
    SafeExecute([&]() {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            auto cpp_config = ConvertConfig(config);
            cpp_guard->UpdateConfig(cpp_config);
        }
    });
}

CLoadGuardConfig GetLoadGuardConfig(CSegmentLoadGuard guard) {
    return SafeExecute([&]() -> CLoadGuardConfig {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            auto config = cpp_guard->GetConfig();
            return ConvertConfig(config);
        }
        return {};
    });
}

void SetPhysicalLimits(CSegmentLoadGuard guard, int64_t memory_limit, int64_t disk_limit) {
    SafeExecute([&]() {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            cpp_guard->SetPhysicalLimits(memory_limit, disk_limit);
        }
    });
}

CResourceUsage GetPhysicalLimits(CSegmentLoadGuard guard) {
    return SafeExecute([&]() -> CResourceUsage {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            auto [memory_limit, disk_limit] = cpp_guard->GetPhysicalLimits();
            return {memory_limit, disk_limit};
        }
        return {};
    });
}

bool CanLoadSegment(CSegmentLoadGuard guard, CSegmentInfo segment_info) {
    return SafeExecute([&]() -> bool {
        if (!guard) return false;
        
        auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
        
        // Create cells for the segment
        std::vector<std::shared_ptr<Cell>> cells;
        if (segment_info.evictable_size > 0) {
            auto evictable_cell = std::make_shared<Cell>(segment_info.evictable_size, true, segment_info.segment_uid, 1);
            cells.push_back(evictable_cell);
        }
        if (segment_info.inevictable_size > 0) {
            auto inevictable_cell = std::make_shared<Cell>(segment_info.inevictable_size, false, segment_info.segment_uid, 2);
            cells.push_back(inevictable_cell);
        }
        
        Segment segment(segment_info.meta_size, std::move(cells));
        return cpp_guard->CanLoadSegment(segment);
    });
}

bool LoadSegment(CSegmentLoadGuard guard, CSegmentInfo segment_info) {
    return SafeExecute([&]() -> bool {
        if (!guard) return false;
        
        auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
        
        // Create cells for the segment
        std::vector<std::shared_ptr<Cell>> cells;
        if (segment_info.evictable_size > 0) {
            auto evictable_cell = std::make_shared<Cell>(segment_info.evictable_size, true, segment_info.segment_uid, 1);
            cells.push_back(evictable_cell);
        }
        if (segment_info.inevictable_size > 0) {
            auto inevictable_cell = std::make_shared<Cell>(segment_info.inevictable_size, false, segment_info.segment_uid, 2);
            cells.push_back(inevictable_cell);
        }
        
        auto segment = std::make_shared<Segment>(segment_info.meta_size, std::move(cells));
        return cpp_guard->LoadSegment(segment);
    });
}

void UnloadSegment(CSegmentLoadGuard guard, int64_t segment_uid) {
    SafeExecute([&]() {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            cpp_guard->UnloadSegment(segment_uid);
        }
    });
}

bool CanLoadCell(CSegmentLoadGuard guard, CCellInfo cell_info) {
    return SafeExecute([&]() -> bool {
        if (!guard) return false;
        
        auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
        Cell cell(cell_info.size, cell_info.evictable, cell_info.segment_uid, cell_info.cell_cid);
        return cpp_guard->CanLoadCell(cell);
    });
}

bool LoadCell(CSegmentLoadGuard guard, CCellInfo cell_info) {
    return SafeExecute([&]() -> bool {
        if (!guard) return false;
        
        auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
        auto cell = std::make_shared<Cell>(cell_info.size, cell_info.evictable, cell_info.segment_uid, cell_info.cell_cid);
        return cpp_guard->LoadCell(cell);
    });
}

void UnloadCell(CSegmentLoadGuard guard, int64_t segment_uid, int64_t cell_cid) {
    SafeExecute([&]() {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            cpp_guard->UnloadCell(segment_uid, cell_cid);
        }
    });
}

CResourceUsage GetShallowUsage(CSegmentLoadGuard guard) {
    return SafeExecute([&]() -> CResourceUsage {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            auto usage = cpp_guard->GetShallowUsage();
            return ConvertResourceUsage(usage);
        }
        return {};
    });
}

CResourceUsage GetDeepUsage(CSegmentLoadGuard guard) {
    return SafeExecute([&]() -> CResourceUsage {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            auto usage = cpp_guard->GetDeepUsage();
            return ConvertResourceUsage(usage);
        }
        return {};
    });
}

CResourceUsage GetPhysicalUsage(CSegmentLoadGuard guard) {
    return SafeExecute([&]() -> CResourceUsage {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            auto usage = cpp_guard->GetPhysicalUsage();
            return ConvertResourceUsage(usage);
        }
        return {};
    });
}

CResourceUsage GetAvailableResources(CSegmentLoadGuard guard) {
    return SafeExecute([&]() -> CResourceUsage {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            auto usage = cpp_guard->GetAvailableResources();
            return ConvertResourceUsage(usage);
        }
        return {};
    });
}

bool TryEvict(CSegmentLoadGuard guard, int64_t target_memory, int64_t target_disk) {
    return SafeExecute([&]() -> bool {
        if (!guard) return false;
        
        auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
        return cpp_guard->TryEvict(target_memory, target_disk);
    });
}

void ForceEviction(CSegmentLoadGuard guard) {
    SafeExecute([&]() {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            cpp_guard->ForceEviction();
        }
    });
}

int64_t GetLoadedSegmentCount(CSegmentLoadGuard guard) {
    return SafeExecute([&]() -> int64_t {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            return static_cast<int64_t>(cpp_guard->GetLoadedSegmentCount());
        }
        return 0;
    });
}

int64_t GetLoadedCellCount(CSegmentLoadGuard guard) {
    return SafeExecute([&]() -> int64_t {
        if (guard) {
            auto cpp_guard = static_cast<SegmentLoadGuard*>(guard);
            return static_cast<int64_t>(cpp_guard->GetLoadedCellCount());
        }
        return 0;
    });
}

const char* GetLastError() {
    return last_error_.c_str();
}

}  // extern "C"