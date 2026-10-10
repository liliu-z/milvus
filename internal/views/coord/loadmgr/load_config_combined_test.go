package loadmgr

import (
	"context"
	"os"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	clientv3 "go.etcd.io/etcd/client/v3"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/metastore/kv/querycoord"
	"github.com/milvus-io/milvus/pkg/v3/kv"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

type combinedTestCatalog struct {
	*querycoord.Catalog
	save func(context.Context) error
}

func (c combinedTestCatalog) SaveLoadConfig(ctx context.Context, _ *querypb.CollectionLoadInfo, _ []*querypb.PartitionLoadInfo, _ []*querypb.Replica) error {
	return c.save(ctx)
}

func TestCombinedPutPublishesOnlyAfterPersistence(t *testing.T) {
	store, _ := newTestStore(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered := make(chan struct{})
	store.catalog = combinedTestCatalog{save: func(ctx context.Context) error {
		close(entered)
		<-ctx.Done()
		return ctx.Err()
	}}
	var notifications atomic.Int64
	store.RegisterObserver(func(int64, bool) { notifications.Add(1) })
	before := store.Get(100)
	done := make(chan error, 1)
	go func() { done <- store.Put(ctx, sampleConfig()) }()
	<-entered
	require.Equal(t, before, store.Get(100))
	require.Zero(t, notifications.Load())
	cancel()
	require.ErrorIs(t, <-done, context.Canceled)
	require.Equal(t, before, store.Get(100))
	require.Zero(t, notifications.Load())
	store.catalog = combinedTestCatalog{save: func(context.Context) error { return nil }}
	require.NoError(t, store.Put(context.Background(), sampleConfig()))
	require.NotNil(t, store.Get(100).Config)
	require.EqualValues(t, 1, notifications.Load())
}

type ambiguousCommitKV struct {
	kv.MetaKv
	ambiguous bool
}

func (k *ambiguousCommitKV) MultiSave(ctx context.Context, values map[string]string) error {
	if err := k.MetaKv.MultiSave(ctx, values); err != nil {
		return err
	}
	if k.ambiguous {
		k.ambiguous = false
		return context.DeadlineExceeded
	}
	return nil
}

func TestCombinedLoadConfigEtcdRecoveryAfterAmbiguousCommit(t *testing.T) {
	endpoint := os.Getenv("COLD_SEARCH_TEST_ETCD")
	if endpoint == "" {
		t.Skip("set COLD_SEARCH_TEST_ETCD to run against an isolated etcd prefix")
	}
	paramtable.Init()
	cli, err := clientv3.New(clientv3.Config{Endpoints: []string{endpoint}, DialTimeout: 5 * time.Second})
	require.NoError(t, err)
	t.Cleanup(func() { cli.Close() })
	prefix := "/cold-search-test/load-config-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	t.Cleanup(func() {
		_, err := cli.Delete(context.Background(), prefix, clientv3.WithPrefix())
		require.NoError(t, err)
	})
	meta := &ambiguousCommitKV{MetaKv: etcdkv.NewEtcdKV(cli, prefix), ambiguous: true}
	catalog := querycoord.NewCatalog(meta)
	var _ loadConfigSaver = catalog
	store, err := RecoverLoadConfigStore(context.Background(), catalog)
	require.NoError(t, err)
	cfg := sampleConfig()
	require.ErrorIs(t, store.Put(context.Background(), cfg), context.DeadlineExceeded)
	require.Nil(t, store.Get(cfg.CollectionID).Config)
	data, err := cli.Get(context.Background(), prefix, clientv3.WithPrefix())
	require.NoError(t, err)
	require.Len(t, data.Kvs, 5)
	for _, entry := range data.Kvs {
		require.Equal(t, data.Kvs[0].ModRevision, entry.ModRevision, "all keys must commit in the same transaction")
	}
	recovered, err := RecoverLoadConfigStore(context.Background(), catalog)
	require.NoError(t, err)
	require.Equal(t, cfg, recovered.Get(cfg.CollectionID).Config)
	require.NoError(t, store.Put(context.Background(), cfg))
	require.Equal(t, cfg, store.Get(cfg.CollectionID).Config)
	// Existing orphan removal remains effective after the combined write.
	next := cfg.Clone()
	next.PartitionIDs = next.PartitionIDs[:1]
	next.Replicas = next.Replicas[:1]
	require.NoError(t, store.Put(context.Background(), next))
	recovered, err = RecoverLoadConfigStore(context.Background(), catalog)
	require.NoError(t, err)
	require.Equal(t, next, recovered.Get(cfg.CollectionID).Config)
}
