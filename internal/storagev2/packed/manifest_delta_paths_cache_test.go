// Copyright 2026 Zilliz
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
	"path/filepath"
	"slices"
	"testing"

	"github.com/bytedance/mockey"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func deltaPathsCacheCount(t *testing.T, result string) float64 {
	t.Helper()
	var metric dto.Metric
	require.NoError(t, manifestDeltaPathsCacheRequests.WithLabelValues(result).Write(&metric))
	return metric.GetCounter().GetValue()
}

func enableManifestMetadataCache(t *testing.T) {
	t.Helper()
	paramtable.Init()
	key := paramtable.Get().CommonCfg.ManifestStatsCacheEnabled.Key
	require.NoError(t, paramtable.Get().Save(key, "true"))
	t.Cleanup(func() { paramtable.Get().Reset(key) })
}

func TestLoadManifestDeltaPathsVersionsAndOwnership(t *testing.T) {
	enableManifestMetadataCache(t)
	cfg := manifestTestStorageConfig(t)
	base := filepath.Join(cfg.RootPath, "delta-cache")
	initial := createBaseManifest(t, base, cfg)
	misses, hits := deltaPathsCacheCount(t, "miss"), deltaPathsCacheCount(t, "hit")
	for range 2 {
		paths, err := GetLoadDeltaLogPathsFromManifest(initial, cfg)
		require.NoError(t, err)
		require.Nil(t, paths)
	}
	require.Equal(t, misses+1, deltaPathsCacheCount(t, "miss"))
	require.Equal(t, hits+1, deltaPathsCacheCount(t, "hit"))
	updated, err := AddDeltaLogsToManifest(initial, cfg, []DeltaLogEntry{
		{Path: filepath.Join(base, "_delta/real"), NumEntries: 2},
		{Path: filepath.Join(base, "_delta/marker"), NumEntries: 0},
	})
	require.NoError(t, err)
	require.NotEqual(t, initial, updated)
	want, err := GetDeltaLogPathsFromManifest(updated, cfg)
	require.NoError(t, err)
	require.Len(t, want, 1, "zero-entry markers must not become file reads")
	for range 2 {
		got, err := GetLoadDeltaLogPathsFromManifest(updated, cfg)
		require.NoError(t, err)
		require.Equal(t, want, got)
		got[0] = "caller mutation"
	}
	old, err := GetLoadDeltaLogPathsFromManifest(initial, cfg)
	require.NoError(t, err)
	require.Nil(t, old, "a later manifest must not change the cached old version")
	// A hit must not call the original reader; disabling must use it again.
	patch := mockey.Mock(GetDeltaLogPathsFromManifest).Return(nil, context.DeadlineExceeded).Build()
	defer patch.UnPatch()
	got, err := GetLoadDeltaLogPathsFromManifest(updated, cfg)
	require.NoError(t, err)
	require.Equal(t, want, got)
	require.NoError(t, paramtable.Get().Save(paramtable.Get().CommonCfg.ManifestStatsCacheEnabled.Key, "false"))
	_, err = GetLoadDeltaLogPathsFromManifest(updated, cfg)
	require.Equal(t, context.DeadlineExceeded, err)
}

func TestLoadManifestDeltaPathsFailureAndBypass(t *testing.T) {
	enableManifestMetadataCache(t)
	cfg := manifestTestStorageConfig(t)
	base := filepath.Join(cfg.RootPath, "missing-then-created")
	missing := MarshalManifestPath(base, 1)
	misses := deltaPathsCacheCount(t, "miss")
	for range 2 {
		_, err := GetLoadDeltaLogPathsFromManifest(missing, cfg)
		require.Error(t, err)
	}
	require.Equal(t, misses+2, deltaPathsCacheCount(t, "miss"))
	created := createBaseManifest(t, base, cfg)
	require.Equal(t, missing, created)
	for range 2 {
		paths, err := GetLoadDeltaLogPathsFromManifest(created, cfg)
		require.NoError(t, err)
		require.Empty(t, paths)
	}
	patch := mockey.Mock(GetDeltaLogPathsFromManifest).Return(nil, context.Canceled).Build()
	defer patch.UnPatch()
	for _, path := range []string{"invalid", MarshalManifestPath(base, -1), MarshalManifestPath(base, 0)} {
		_, err := GetLoadDeltaLogPathsFromManifest(path, cfg)
		require.Equal(t, context.Canceled, err)
	}
	_, err := GetLoadDeltaLogPathsFromManifest(created, nil)
	require.Equal(t, context.Canceled, err)
	other := proto.Clone(cfg).(*indexpb.StorageConfig)
	other.RootPath += "-different"
	_, err = GetLoadDeltaLogPathsFromManifest(created, other)
	require.Equal(t, context.Canceled, err, "storage identity must isolate a cached path")
}

func TestLoadManifestDeltaPathsCapacityAndErrorIdentity(t *testing.T) {
	cfg := &indexpb.StorageConfig{StorageType: "local"}
	key, ok := makeManifestStatsKey(MarshalManifestPath("exact//segment", 1), cfg)
	require.True(t, ok)
	want := []string{"exact//../delta"}
	c := newManifestMetadataCache(2, manifestDeltaPathsBytes(key, want), slices.Clone[[]string], manifestDeltaPathsBytes)
	c.put(key, want)
	want[0] = "mutated input"
	got, found := c.get(key)
	require.True(t, found)
	require.Equal(t, []string{"exact//../delta"}, got)
	c.put(manifestStatsKey{path: "oversize"}, []string{string(make([]byte, c.maxBytes+1))})
	require.Len(t, c.entries, 1)
	for range 2 {
		_, outcome, err := c.read(MarshalManifestPath("failure", 1), cfg, func() ([]string, error) {
			return nil, context.DeadlineExceeded
		})
		require.Equal(t, "miss", outcome)
		require.Equal(t, context.DeadlineExceeded, err)
	}
}
