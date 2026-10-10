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

package segments

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/mocks/util/mock_segcore"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/initcore"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// Exercise real manifests, real Parquet reads, parent/child delete merging and
// a missing delete file after the path metadata has already been cached.
func TestLoadDeltalogsMetadataCacheReadsFilesAndVersions(t *testing.T) {
	paramtable.Init()
	initcore.InitExecExpressionFunctionFactory()
	pt := paramtable.Get()
	root := t.TempDir()
	for key, value := range map[string]string{
		pt.CommonCfg.StorageType.Key: "local", pt.LocalStorageCfg.Path.Key: root,
		pt.CommonCfg.ManifestStatsCacheEnabled.Key: "true",
	} {
		require.NoError(t, pt.Save(key, value))
		t.Cleanup(func() { pt.Reset(key) })
	}
	cfg := createStorageConfig()
	manager := NewManager()
	const collectionID = int64(90001)
	schema := mock_segcore.GenTestCollectionSchema("delta-cache", schemapb.DataType_Int64, false)
	require.NoError(t, manager.Collection.PutOrRef(collectionID, schema, nil, &querypb.LoadMetaInfo{
		LoadType: querypb.LoadType_LoadCollection, CollectionID: collectionID,
	}))
	defer manager.Collection.Unref(collectionID, 1)
	loader := &segmentLoader{manager: manager}
	writeDelta := func(path string, pk, timestamp int64) {
		t.Helper()
		require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		as := arrow.NewSchema([]arrow.Field{{Name: "0", Type: arrow.PrimitiveTypes.Int64}, {Name: "1", Type: arrow.PrimitiveTypes.Int64}}, nil)
		builder := array.NewRecordBuilder(memory.DefaultAllocator, as)
		defer builder.Release()
		builder.Field(0).(*array.Int64Builder).Append(pk)
		builder.Field(1).(*array.Int64Builder).Append(timestamp)
		record := builder.NewRecord()
		defer record.Release()
		writer, err := storage.NewDeltalogWriter(context.Background(), collectionID, 90002, 90003, 1,
			schemapb.DataType_Int64, path, storage.WithVersion(storage.StorageV2), storage.WithStorageConfig(cfg))
		require.NoError(t, err)
		require.NoError(t, writer.Write(storage.NewSimpleArrowRecord(record, map[int64]int{0: 0, 1: 1})))
		require.NoError(t, writer.Close())
	}
	parentFile, childFile := filepath.Join(root, "parent/_delta/one.parquet"), filepath.Join(root, "child/_delta/one.parquet")
	writeDelta(parentFile, 11, 101)
	writeDelta(childFile, 22, 202)
	manifest := func(base, file string) string {
		t.Helper()
		path, err := packed.CommitManifestUpdates(base, packed.ManifestEarliest, cfg, &packed.ManifestUpdates{
			DeltaLogs: []packed.DeltaLogEntry{{Path: file, NumEntries: 1}},
		})
		require.NoError(t, err)
		return path
	}
	parent := manifest(filepath.Join(root, "parent"), parentFile)
	child := manifest(filepath.Join(root, "child"), childFile)
	load := func(parent string) (*storage.DeltaData, error) {
		segment := &deltaLoadTestSegment{id: 90003, collectionID: collectionID}
		err := loader.loadDeltalogs(context.Background(), segment, &querypb.SegmentLoadInfo{
			CollectionID: collectionID, SegmentID: segment.id, ManifestPath: parent,
			ChildManifestPaths: []string{child},
		})
		return segment.deltaData, err
	}
	for range 2 {
		data, err := load(parent)
		require.NoError(t, err)
		require.EqualValues(t, 2, data.DeleteRowCount())
		require.Equal(t, []uint64{101, 202}, data.DeleteTimestamps())
		require.Equal(t, storage.NewInt64PrimaryKey(11), data.DeletePks().Get(0))
		require.Equal(t, storage.NewInt64PrimaryKey(22), data.DeletePks().Get(1))
	}
	newFile := filepath.Join(root, "parent/_delta/two.parquet")
	writeDelta(newFile, 33, 303)
	updated, err := packed.AddDeltaLogsToManifest(parent, cfg, []packed.DeltaLogEntry{{Path: newFile, NumEntries: 1}})
	require.NoError(t, err)
	data, err := load(updated)
	require.NoError(t, err)
	require.EqualValues(t, 3, data.DeleteRowCount())
	require.Equal(t, []uint64{101, 303, 202}, data.DeleteTimestamps())
	data, err = load(parent)
	require.NoError(t, err)
	require.EqualValues(t, 2, data.DeleteRowCount(), "old version must remain independent")
	require.NoError(t, os.Remove(parentFile))
	data, err = load(updated)
	require.Error(t, err, "a cached path must still open the actual delete file")
	require.Nil(t, data, "a partial delete load must not be applied")
	writeDelta(parentFile, 11, 101)
	data, err = load(updated)
	require.NoError(t, err, "file-read failure must not poison a subsequent load")
	require.EqualValues(t, 3, data.DeleteRowCount())
}
