package querynodev2

import (
	"context"

	"github.com/your-project/segments"
	"github.com/your-project/paramtable"
	"github.com/your-project/scheduler"
)

func NewQueryNode(ctx context.Context, node *QueryNode) (*QueryNode, error) {
	if err := node.init(); err != nil {
		return nil, err
	}
	manager := segments.NewManager(
		paramtable.Get().QueryNodeCfg.MmapDirPath.GetValue(),
		node.chunkManager,
	)
	loader := segments.NewLoader(ctx, manager, node.chunkManager, node.loadGuard)
	manager.SetLoader(loader)

	s, err := scheduler.NewScheduler(ctx)
	if err != nil {
		return nil, err
	}
	node.scheduler = s

	return node, nil
} 