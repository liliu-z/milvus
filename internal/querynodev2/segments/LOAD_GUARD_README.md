# Load Guard 设计与实现

## 概述

Load Guard 是 Milvus QueryNode 的新资源管理系统，旨在解决引入 CachingLayer 后的资源管理问题。它提供了更精细的资源控制，支持浅加载和深加载的准入机制，以及与驱逐策略的集成。

## 核心概念

### Cell（数据单元）
- 代表可以被加载到内存/磁盘的数据片段
- 每个 Cell 有唯一ID、大小估算、是否可驱逐的属性
- 支持深加载状态跟踪

### SegmentInfo（段信息）
- 包含段的元数据大小、可驱逐数据大小、不可驱逐数据大小
- 管理段内的所有 Cell
- 跟踪浅加载状态

### SegmentLoadGuard（加载守护）
- 核心资源管理器
- 实现浅加载和深加载的准入控制
- 支持逻辑水位和物理水位双重保护
- 集成驱逐机制

## 设计特性

### 1. 两级准入机制

#### 浅加载准入（Segment Level）
- 当段被标记为加载时进行检查
- 基于逻辑资源使用量和缓存比例
- 考虑元数据、不可驱逐数据、以及可驱逐数据的预期缓存量

#### 深加载准入（Cell Level）
- 当数据实际需要加载到内存时进行检查
- 同时检查逻辑水位和物理水位
- 在资源不足时尝试驱逐

### 2. 资源分类管理

#### 浅加载资源（Shallow Resources）
- 已被标记为加载但实际数据可能在远程存储的段
- `shallowEvictable`: 可驱逐数据的总大小
- `shallowInevictable`: 不可驱逐数据的总大小

#### 深加载资源（Deep Resources）
- 实际加载到本地内存/磁盘的数据
- `deepEvictable`: 已加载的可驱逐数据大小
- `deepInevictable`: 已加载的不可驱逐数据大小
- `deepLoading`: 正在加载中的数据大小

### 3. 缓存比例控制

通过 `cache_ratio` 参数控制为可驱逐数据预留多少资源：
- `cache_ratio = 1`: 关闭驱逐时，所有数据都需要预留空间
- `cache_ratio < 1`: 启用驱逐时，只为部分可驱逐数据预留空间
- 这允许加载超过物理容量的数据，依靠驱逐机制管理

### 4. 水位线管理

#### 逻辑水位
- 基于估算的资源使用量
- 用于准入控制和驱逐触发

#### 物理水位
- 基于实际的内存/磁盘使用量
- 作为最终的安全保障

#### 三级水位配置
- `Low`: 驱逐目标水位
- `High`: 驱逐触发水位  
- `Max` (OverloadPercentage): 绝对上限

## 配置参数

```go
type LoadGuardConfig struct {
    OverloadPercentage float64 // 最大资源使用百分比 (默认: 0.9)
    CacheRatio         float64 // 缓存比例 (默认: 0.2)
    EvictionEnabled    bool    // 是否启用驱逐 (默认: true)
    LowWatermark       float64 // 低水位 (默认: 0.7)
    HighWatermark      float64 // 高水位 (默认: 0.8)
    EvictionInterval   time.Duration // 驱逐检查间隔 (默认: 3s)
}
```

## 使用场景

### 场景1: 关闭驱逐 + 同步预热
- `EvictionEnabled = false`
- `WarmupPolicy = sync`
- 行为类似原有系统，严格限制加载数据量

### 场景2: 关闭驱逐 + 禁用预热
- `EvictionEnabled = false` 
- `WarmupPolicy = disable`
- 允许浅加载超量数据，但会在访问时 OOM

### 场景3: 启用驱逐 + 同步预热
- `EvictionEnabled = true`
- `WarmupPolicy = sync`
- 可能因为保守的准入策略限制加载能力

### 场景4: 启用驱逐 + 禁用预热（推荐）
- `EvictionEnabled = true`
- `WarmupPolicy = disable`
- 最大化资源利用率，通过驱逐避免 OOM

## 集成方式

### SegmentLoader 集成
```go
// 创建 LoadGuard
config := DefaultLoadGuardConfig()
loadGuard, err := NewSegmentLoadGuard(config)

// 浅加载准入
segmentInfo, err := SegmentInfoFromLoadInfo(schema, loadInfo)
err = loadGuard.SegmentLoadAdmission(ctx, segmentInfo)

// 深加载准入
err = loadGuard.CellLoadAdmission(ctx, cell)
loadGuard.OnCellLoaded(cell) // 加载成功后通知
```

### 与 CachingLayer 集成
- LoadGuard 提供准入控制接口
- CachingLayer 在加载数据前调用 `CellLoadAdmission`
- 数据加载完成后调用 `OnCellLoaded`
- 驱逐时调用 `OnCellEvicted`

## 优势

1. **精细化资源管理**: 区分可驱逐和不可驱逐数据
2. **弹性容量**: 支持加载超过物理容量的数据
3. **安全保障**: 逻辑和物理双重水位保护
4. **热数据保护**: 通过缓存比例预留热数据空间
5. **渐进式部署**: 可以与现有系统并存，逐步迁移

## 监控指标

LoadGuard 提供丰富的统计信息：
```go
stats := loadGuard.GetStats()
// stats 包含:
// - limit: 资源限制
// - shallow_evictable/shallow_inevictable: 浅加载资源使用
// - deep_evictable/deep_inevictable: 深加载资源使用
// - deep_loading: 正在加载的资源
// - shallow_usage/deep_usage: 计算的使用量
// - physical_usage: 物理使用量
```

## 未来扩展

1. **更精确的资源估算**: 集成实际加载后的大小反馈
2. **智能驱逐策略**: 基于访问模式的 LRU/LFU 驱逐
3. **多资源类型支持**: 扩展到 GPU 内存等其他资源
4. **动态配置调整**: 支持运行时调整配置参数
5. **预测性加载**: 基于查询模式预测数据需求

## 测试

运行测试验证实现：
```bash
go test ./internal/querynodev2/segments -v -run TestLoadGuard
```

测试覆盖：
- 配置验证
- Cell 和 SegmentInfo 基本功能
- 准入控制逻辑
- 驱逐机制
- 内存监控 