package querycoord

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	kvmocks "github.com/milvus-io/milvus/internal/kv/mocks"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestSaveLoadConfigMatchesExistingSerialization(t *testing.T) {
	paramtable.Init()
	collection := &querypb.CollectionLoadInfo{CollectionID: 7, ReplicaNumber: 1, LoadFields: []int64{100, 101}}
	partitions := []*querypb.PartitionLoadInfo{{CollectionID: 7, PartitionID: 8}}
	replicas := []*querypb.Replica{{CollectionID: 7, ID: 9, ResourceGroup: "default"}}
	for _, mode := range []string{"combined", "operation_limit", "byte_limit", "empty_replicas"} {
		t.Run(mode, func(t *testing.T) {
			r := proto.Clone(replicas[0]).(*querypb.Replica)
			rs := []*querypb.Replica{r}
			if mode == "operation_limit" {
				require.NoError(t, paramtable.Get().Save("metastore.maxEtcdTxnNum", "2"))
				t.Cleanup(func() { paramtable.Get().Reset("metastore.maxEtcdTxnNum") })
			}
			if mode == "byte_limit" {
				r.ResourceGroup = strings.Repeat("a", 65537)
			}
			if mode == "empty_replicas" {
				rs = nil
			}
			oldKV := kvmocks.NewMetaKv(t)
			expected := map[string]string{}
			oldKV.EXPECT().MultiSave(mock.Anything, mock.Anything).Run(func(_ context.Context, values map[string]string) {
				for key, value := range values {
					expected[key] = value
				}
			}).Return(nil)
			old := Catalog{cli: oldKV}
			require.NoError(t, old.SaveCollection(context.Background(), collection, partitions...))
			if len(rs) > 0 {
				require.NoError(t, old.SaveReplica(context.Background(), rs...))
			}

			newKV := kvmocks.NewMetaKv(t)
			actual := map[string]string{}
			calls := 1
			if mode == "operation_limit" || mode == "byte_limit" {
				calls = 2
			}
			newKV.EXPECT().MultiSave(mock.Anything, mock.Anything).Run(func(_ context.Context, values map[string]string) {
				if len(actual) == 0 {
					require.Contains(t, values, EncodeCollectionLoadInfoKey(7))
				}
				for key, value := range values {
					actual[key] = value
				}
			}).Return(nil).Times(calls)
			require.NoError(t, (Catalog{cli: newKV}).SaveLoadConfig(context.Background(), collection, partitions, rs))
			require.Equal(t, expected, actual)
		})
	}
}

func TestSaveLoadConfigDoesNotSwallowPersistenceFailure(t *testing.T) {
	paramtable.Init()
	for _, limit := range []string{"64", "1"} {
		t.Run(limit, func(t *testing.T) {
			require.NoError(t, paramtable.Get().Save("metastore.maxEtcdTxnNum", limit))
			t.Cleanup(func() { paramtable.Get().Reset("metastore.maxEtcdTxnNum") })
			kv := kvmocks.NewMetaKv(t)
			kv.EXPECT().MultiSave(mock.Anything, mock.Anything).Return(context.DeadlineExceeded).Once()
			err := (Catalog{cli: kv}).SaveLoadConfig(context.Background(), &querypb.CollectionLoadInfo{CollectionID: 7}, nil, []*querypb.Replica{{ID: 8, CollectionID: 7}})
			require.ErrorIs(t, err, context.DeadlineExceeded)
		})
	}
}
