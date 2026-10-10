// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package packed

import (
	"container/list"
	"crypto/sha256"
	"maps"
	"slices"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// These are immutable manifest descriptions, never segment data or stat-file
// contents. Native storage already caches the corresponding immutable manifests;
// this avoids repeating their C/Go conversion for each load phase.
var loadManifestStatsCache = newManifestStatsCache(128, 16<<20)

// Delta paths are metadata too. The delete files themselves remain uncached and
// are opened by each load, including compact-to child-manifest overlays.
var loadManifestDeltaPathsCache = newManifestMetadataCache(128, 1<<20, slices.Clone[[]string], manifestDeltaPathsBytes)

var (
	manifestStatsCacheMetricsOnce sync.Once
	manifestStatsCacheRequests    = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "milvus_storage_manifest_stats_cache_requests_total",
		Help: "Immutable load-manifest stats cache lookups by outcome.",
	}, []string{"result"})
	manifestDeltaPathsCacheRequests = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "milvus_storage_manifest_delta_paths_cache_requests_total",
		Help: "Immutable load-manifest delta-path cache lookups by outcome.",
	}, []string{"result"})
)

type manifestStatsKey struct {
	path    string
	storage [sha256.Size]byte
}

type manifestMetadataEntry[V any] struct {
	key   manifestStatsKey
	value V
	bytes int
}

type manifestMetadataCache[V any] struct {
	mu                          sync.Mutex
	entries                     map[manifestStatsKey]*list.Element
	lru                         *list.List
	maxEntries, maxBytes, bytes int
	clone                       func(V) V
	size                        func(manifestStatsKey, V) int
}

func newManifestMetadataCache[V any](entries, bytes int, clone func(V) V, size func(manifestStatsKey, V) int) *manifestMetadataCache[V] {
	return &manifestMetadataCache[V]{
		entries: make(map[manifestStatsKey]*list.Element), lru: list.New(),
		maxEntries: entries, maxBytes: bytes,
		clone: clone, size: size,
	}
}

func newManifestStatsCache(entries, bytes int) *manifestMetadataCache[map[string]ManifestStat] {
	return newManifestMetadataCache(entries, bytes, cloneManifestStats, manifestStatsBytes)
}

func makeManifestStatsKey(path string, storage *indexpb.StorageConfig) (manifestStatsKey, bool) {
	base, version, err := UnmarshalManifestPath(path)
	// Latest-version lookups and default/implicit filesystem configuration are
	// not stable cache identities. Preserve their ordinary read/error path.
	if err != nil || base == "" || version <= 0 || storage == nil {
		return manifestStatsKey{}, false
	}
	encoded, err := (proto.MarshalOptions{Deterministic: true}).Marshal(storage)
	if err != nil {
		return manifestStatsKey{}, false
	}
	return manifestStatsKey{path: path, storage: sha256.Sum256(encoded)}, true
}

func cloneManifestStats(stats map[string]ManifestStat) map[string]ManifestStat {
	cloned := maps.Clone(stats)
	for key, value := range cloned {
		value.Paths = slices.Clone(value.Paths)
		value.Metadata = maps.Clone(value.Metadata)
		cloned[key] = value
	}
	return cloned
}

// Bound both entry count and accounted bytes (strings plus conservative
// container allowances). This is not an allocator-exact heap measurement.
func manifestStatsBytes(key manifestStatsKey, stats map[string]ManifestStat) int {
	n := 256 + len(key.path)
	for name, stat := range stats {
		n += 256 + len(name)
		for _, path := range stat.Paths {
			n += 32 + len(path)
		}
		for name, value := range stat.Metadata {
			n += 128 + len(name) + len(value)
		}
	}
	return n
}

func (c *manifestMetadataCache[V]) get(key manifestStatsKey) (V, bool) {
	c.mu.Lock()
	entry, found := c.entries[key]
	if !found {
		c.mu.Unlock()
		var zero V
		return zero, false
	}
	c.lru.MoveToFront(entry)
	value := entry.Value.(*manifestMetadataEntry[V]).value
	c.mu.Unlock()
	// Published entries are immutable. Readers own all returned maps/slices.
	return c.clone(value), true
}

func (c *manifestMetadataCache[V]) put(key manifestStatsKey, value V) {
	bytes := c.size(key, value)
	if c.maxEntries <= 0 || bytes > c.maxBytes {
		return
	}
	entry := &manifestMetadataEntry[V]{key: key, value: c.clone(value), bytes: bytes}
	c.mu.Lock()
	defer c.mu.Unlock()
	if old, found := c.entries[key]; found {
		c.lru.MoveToFront(old)
		return
	}
	for len(c.entries) >= c.maxEntries || c.bytes+bytes > c.maxBytes {
		old := c.lru.Back()
		value := old.Value.(*manifestMetadataEntry[V])
		delete(c.entries, value.key)
		c.bytes -= value.bytes
		c.lru.Remove(old)
	}
	c.entries[key] = c.lru.PushFront(entry)
	c.bytes += bytes
}

func (c *manifestMetadataCache[V]) read(path string, storage *indexpb.StorageConfig,
	read func() (V, error),
) (V, string, error) {
	key, cacheable := makeManifestStatsKey(path, storage)
	if !cacheable {
		stats, err := read()
		return stats, "bypass", err
	}
	if stats, found := c.get(key); found {
		return stats, "hit", nil
	}
	// No lock is held over IO. Concurrent misses may perform duplicate reads;
	// errors are never cached and retain the original caller's error handling.
	stats, err := read()
	if err == nil {
		c.put(key, stats)
	}
	return stats, "miss", err
}

func getCachedLoadManifestStats(path string, storage *indexpb.StorageConfig, origin ManifestReadOrigin) (map[string]ManifestStat, error) {
	registerManifestMetadataCacheMetrics()
	stats, result, err := loadManifestStatsCache.read(path, storage, func() (map[string]ManifestStat, error) {
		return getManifestStats(path, storage, origin)
	})
	manifestStatsCacheRequests.WithLabelValues(result).Inc()
	return stats, err
}

func registerManifestMetadataCacheMetrics() {
	manifestStatsCacheMetricsOnce.Do(func() {
		metrics.GetRegisterer().MustRegister(manifestStatsCacheRequests, manifestDeltaPathsCacheRequests)
	})
}

func manifestDeltaPathsBytes(key manifestStatsKey, paths []string) int {
	n := 256 + len(key.path)
	for _, path := range paths {
		n += 32 + len(path)
	}
	return n
}

// GetLoadDeltaLogPathsFromManifest is the ordinary, non-extfs load reader with
// optional reuse of immutable delta-path descriptions. Keep the original reader
// as the miss/bypass path, including marker filtering and error propagation.
func GetLoadDeltaLogPathsFromManifest(path string, storage *indexpb.StorageConfig) ([]string, error) {
	if !paramtable.Get().CommonCfg.ManifestStatsCacheEnabled.GetAsBool() {
		return GetDeltaLogPathsFromManifest(path, storage)
	}
	registerManifestMetadataCacheMetrics()
	paths, result, err := loadManifestDeltaPathsCache.read(path, storage, func() ([]string, error) {
		return GetDeltaLogPathsFromManifest(path, storage)
	})
	manifestDeltaPathsCacheRequests.WithLabelValues(result).Inc()
	return paths, err
}
