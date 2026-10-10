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
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestManifestStatsCacheIsolationAndOwnership(t *testing.T) {
	c := newManifestStatsCache(8, 1<<20)
	cfg := &indexpb.StorageConfig{StorageType: "remote", BucketName: "a", Address: "one", SecretAccessKey: "first"}
	path := MarshalManifestPath("files/a//../segment", 1)
	calls := 0
	read := func() (map[string]ManifestStat, error) {
		calls++
		return map[string]ManifestStat{"bf.100": {Paths: []string{"exact//key"}, Metadata: map[string]string{"version": "1"}}}, nil
	}
	first, outcome, err := c.read(path, cfg, read)
	require.NoError(t, err)
	require.Equal(t, "miss", outcome)
	first["bf.100"].Paths[0] = "mutated"
	first["bf.100"].Metadata["version"] = "mutated"
	delete(first, "bf.100")
	for range 2 {
		got, outcome, err := c.read(path, cfg, read)
		require.NoError(t, err)
		require.Equal(t, "hit", outcome)
		require.Equal(t, "exact//key", got["bf.100"].Paths[0])
		require.Equal(t, "1", got["bf.100"].Metadata["version"])
		got["bf.100"].Paths[0] = "caller mutation"
	}
	require.Equal(t, 1, calls)
	for _, changed := range []string{MarshalManifestPath("files/a//../segment", 2), MarshalManifestPath("files/segment", 1)} {
		_, outcome, err := c.read(changed, cfg, read)
		require.NoError(t, err)
		require.Equal(t, "miss", outcome)
	}
	for _, modify := range []func(*indexpb.StorageConfig){
		func(c *indexpb.StorageConfig) { c.BucketName = "other" },
		func(c *indexpb.StorageConfig) { c.Address = "other" },
		func(c *indexpb.StorageConfig) { c.SecretAccessKey = "rotated" },
		func(c *indexpb.StorageConfig) { c.RootPath = "other" },
	} {
		other := proto.Clone(cfg).(*indexpb.StorageConfig)
		modify(other)
		_, outcome, err := c.read(path, other, read)
		require.NoError(t, err)
		require.Equal(t, "miss", outcome)
	}
}

func TestManifestStatsCacheBypassErrorsAndEmptyResult(t *testing.T) {
	c := newManifestStatsCache(8, 1<<20)
	cfg := &indexpb.StorageConfig{StorageType: "local"}
	calls := 0
	read := func() (map[string]ManifestStat, error) { calls++; return nil, context.DeadlineExceeded }
	path := MarshalManifestPath("base", 1)
	for range 2 {
		_, outcome, err := c.read(path, cfg, read)
		require.Equal(t, context.DeadlineExceeded, err)
		require.Equal(t, "miss", outcome)
	}
	require.Equal(t, 2, calls, "failure must not poison retries")
	for _, value := range []string{"invalid", MarshalManifestPath("base", -1), MarshalManifestPath("base", 0), MarshalManifestPath("", 1)} {
		_, outcome, err := c.read(value, cfg, read)
		require.Equal(t, context.DeadlineExceeded, err)
		require.Equal(t, "bypass", outcome)
	}
	_, outcome, err := c.read(path, nil, read)
	require.Equal(t, context.DeadlineExceeded, err)
	require.Equal(t, "bypass", outcome)
	_, outcome, err = c.read(path, cfg, func() (map[string]ManifestStat, error) { return map[string]ManifestStat{}, nil })
	require.NoError(t, err)
	require.Equal(t, "miss", outcome)
	stats, outcome, err := c.read(path, cfg, read)
	require.NoError(t, err)
	require.Equal(t, "hit", outcome)
	require.Empty(t, stats)
}

func TestManifestStatsCacheBoundsAndConcurrency(t *testing.T) {
	cfg := &indexpb.StorageConfig{StorageType: "local"}
	key := func(i int) manifestStatsKey {
		k, ok := makeManifestStatsKey(MarshalManifestPath(fmt.Sprint(i), 1), cfg)
		require.True(t, ok)
		return k
	}
	c := newManifestStatsCache(2, 1<<20)
	c.put(key(1), nil)
	c.put(key(2), nil)
	_, found := c.get(key(1))
	require.True(t, found)
	c.put(key(3), nil)
	_, found = c.get(key(2))
	require.False(t, found, "least recently used entry must be evicted")
	one := manifestStatsBytes(key(1), nil)
	c = newManifestStatsCache(10, one)
	c.put(key(1), nil)
	c.put(key(2), nil)
	require.Len(t, c.entries, 1)
	require.Equal(t, one, c.bytes)
	c.put(key(3), map[string]ManifestStat{"too large": {Paths: []string{"large"}}})
	require.Len(t, c.entries, 1)
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				path := MarshalManifestPath(fmt.Sprint(i), 1)
				got, _, err := c.read(path, cfg, func() (map[string]ManifestStat, error) { return nil, nil })
				require.NoError(t, err)
				require.Nil(t, got)
			}
		}()
	}
	wg.Wait()
	require.LessOrEqual(t, c.bytes, c.maxBytes)
	require.LessOrEqual(t, len(c.entries), c.maxEntries)
}

func TestManifestStatsCacheRealManifestAndResolver(t *testing.T) {
	paramtable.Init()
	cfg := manifestTestStorageConfig(t)
	path, err := CommitManifestUpdates(filepath.Join(cfg.RootPath, "cached-stats/segment"), ManifestEarliest, cfg, &ManifestUpdates{
		Stats: []StatEntry{{Key: "bloom_filter.100", Files: []string{"exact//bf"}, Metadata: map[string]string{"memory_size": "16"}}},
	})
	require.NoError(t, err)
	key := paramtable.Get().CommonCfg.ManifestStatsCacheEnabled.Key
	require.NoError(t, paramtable.Get().Save(key, "true"))
	t.Cleanup(func() { paramtable.Get().Reset(key) })
	makeResolver := func(path string) *StatsResolver {
		r := NewStatsResolverFromLoadInfo(&querypb.SegmentLoadInfo{ManifestPath: path})
		r.storageConfig = cfg
		return r.WithManifestReadOrigin(ManifestReadCreate)
	}
	before := manifestHistogram(t, "manifest_create", "total", "success").GetSampleCount()
	for range 2 {
		r := makeResolver(path)
		value, err := r.BloomFilterMemorySize(100)
		require.NoError(t, err)
		require.EqualValues(t, 16, value)
	}
	after := manifestHistogram(t, "manifest_create", "total", "success").GetSampleCount()
	require.Equal(t, before+1, after, "independent load phases should perform one FFI extraction")
	newPath, err := AddStatsToManifest(path, cfg, []StatEntry{{Key: "bloom_filter.100", Files: []string{"new/bf"}, Metadata: map[string]string{"memory_size": "32"}}})
	require.NoError(t, err)
	require.NotEqual(t, path, newPath)
	value, err := makeResolver(newPath).BloomFilterMemorySize(100)
	require.NoError(t, err)
	require.EqualValues(t, 32, value)
	value, err = makeResolver(path).BloomFilterMemorySize(100)
	require.NoError(t, err)
	require.EqualValues(t, 16, value, "old immutable version remains isolated")
	require.NoError(t, paramtable.Get().Save(key, "false"))
	_, err = makeResolver(path).BloomFilterMemorySize(100)
	require.NoError(t, err)
	require.Equal(t, after+2, manifestHistogram(t, "manifest_create", "total", "success").GetSampleCount(), "new version and disabled cache each perform FFI")
}

func TestManifestStatsCacheMissingManifestRecovery(t *testing.T) {
	cfg := manifestTestStorageConfig(t)
	base := filepath.Join(cfg.RootPath, "created-after-failed-load")
	path := MarshalManifestPath(base, 1)
	c := newManifestStatsCache(8, 1<<20)
	calls := 0
	read := func() (map[string]ManifestStat, error) {
		calls++
		return getManifestStats(path, cfg, ManifestReadCreate)
	}
	for range 2 {
		_, outcome, err := c.read(path, cfg, read)
		require.Error(t, err)
		require.Equal(t, "miss", outcome)
		require.Empty(t, c.entries, "real FFI failure must not be cached")
	}
	created, err := CommitManifestUpdates(base, ManifestEarliest, cfg, &ManifestUpdates{
		Stats: []StatEntry{{Key: "bloom_filter.100", Files: []string{"recovered/bf"}}},
	})
	require.NoError(t, err)
	require.Equal(t, path, created)
	for _, want := range []string{"miss", "hit"} {
		stats, outcome, err := c.read(path, cfg, read)
		require.NoError(t, err)
		require.Equal(t, want, outcome)
		require.Contains(t, stats, "bloom_filter.100")
	}
	require.Equal(t, 3, calls)
}
