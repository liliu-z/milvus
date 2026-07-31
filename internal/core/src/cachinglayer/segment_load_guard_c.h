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

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for SegmentLoadGuard
typedef void* CSegmentLoadGuard;

// Configuration structure
typedef struct {
    double overload_percentage;
    double cache_ratio;
    bool eviction_enabled;
    double low_watermark;
    double high_watermark;
    int64_t eviction_interval_ms;
} CLoadGuardConfig;

// Resource usage structure
typedef struct {
    int64_t memory_bytes;
    int64_t disk_bytes;
} CResourceUsage;

// Segment information structure
typedef struct {
    int64_t segment_uid;
    int64_t meta_size;
    int64_t evictable_size;
    int64_t inevictable_size;
} CSegmentInfo;

// Cell information structure
typedef struct {
    int64_t segment_uid;
    int64_t cell_cid;
    int64_t size;
    bool evictable;
} CCellInfo;

// SegmentLoadGuard lifecycle
CSegmentLoadGuard NewSegmentLoadGuard(CLoadGuardConfig config);
void DeleteSegmentLoadGuard(CSegmentLoadGuard guard);

// Configuration management
void UpdateLoadGuardConfig(CSegmentLoadGuard guard, CLoadGuardConfig config);
CLoadGuardConfig GetLoadGuardConfig(CSegmentLoadGuard guard);

// Physical resource limits
void SetPhysicalLimits(CSegmentLoadGuard guard, int64_t memory_limit, int64_t disk_limit);
CResourceUsage GetPhysicalLimits(CSegmentLoadGuard guard);

// Segment load admission control
bool CanLoadSegment(CSegmentLoadGuard guard, CSegmentInfo segment_info);
bool LoadSegment(CSegmentLoadGuard guard, CSegmentInfo segment_info);
void UnloadSegment(CSegmentLoadGuard guard, int64_t segment_uid);

// Cell load admission control
bool CanLoadCell(CSegmentLoadGuard guard, CCellInfo cell_info);
bool LoadCell(CSegmentLoadGuard guard, CCellInfo cell_info);
void UnloadCell(CSegmentLoadGuard guard, int64_t segment_uid, int64_t cell_cid);

// Resource usage queries
CResourceUsage GetShallowUsage(CSegmentLoadGuard guard);
CResourceUsage GetDeepUsage(CSegmentLoadGuard guard);
CResourceUsage GetPhysicalUsage(CSegmentLoadGuard guard);
CResourceUsage GetAvailableResources(CSegmentLoadGuard guard);

// Eviction control
bool TryEvict(CSegmentLoadGuard guard, int64_t target_memory, int64_t target_disk);
void ForceEviction(CSegmentLoadGuard guard);

// Status and monitoring
int64_t GetLoadedSegmentCount(CSegmentLoadGuard guard);
int64_t GetLoadedCellCount(CSegmentLoadGuard guard);

// Error handling
const char* GetLastError();

#ifdef __cplusplus
}
#endif