// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
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
	"sync"

	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus/pkg/v2/log"
)

// searchOnSegments performs search on listed segments
// all segment ids are validated before calling this function
func searchSegments(ctx context.Context, mgr *Manager, segments []Segment, segType SegmentType, searchReq *SearchRequest) ([]*SearchResult, error) {
	if len(segments) == 0 {
		return nil, nil
	}

	// Use slice + mutex instead of channel to collect results (fewer allocations)
	results := make([]*SearchResult, len(segments))
	var g errgroup.Group
	for i, segment := range segments {
		idx := i
		seg := segment
		g.Go(func() error {
			if seg.IsLazyLoad() {
				ctx, cancel := withLazyLoadTimeoutContext(ctx)
				defer cancel()
				searcher := func(ctx context.Context, s Segment) error {
					searchResult, err := s.Search(ctx, searchReq)
					if err != nil {
						return err
					}
					results[idx] = searchResult
					return nil
				}
				_, err := mgr.DiskCache.Do(ctx, seg.ID(), searcher)
				if err != nil {
					log.Warn("failed to do search for disk cache", zap.Int64("segID", seg.ID()), zap.Error(err))
				}
				return err
			}
			searchResult, err := seg.Search(ctx, searchReq)
			if err != nil {
				return err
			}
			results[idx] = searchResult
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		DeleteSearchResults(results)
		return nil, err
	}
	return results, nil
}

// searchSegmentsStreamly performs search on listed segments in a stream mode instead of a batch mode
// all segment ids are validated before calling this function
func searchSegmentsStreamly(ctx context.Context,
	mgr *Manager,
	segments []Segment,
	searchReq *SearchRequest,
	streamReduce func(result *SearchResult) error,
) error {
	searchResultsToClear := make([]*SearchResult, 0, len(segments))
	var reduceMutex sync.Mutex
	searcher := func(ctx context.Context, seg Segment) error {
		searchResult, searchErr := seg.Search(ctx, searchReq)
		if searchErr != nil {
			return searchErr
		}
		reduceMutex.Lock()
		searchResultsToClear = append(searchResultsToClear, searchResult)
		reducedErr := streamReduce(searchResult)
		reduceMutex.Unlock()
		return reducedErr
	}

	var g errgroup.Group
	for _, segment := range segments {
		seg := segment
		g.Go(func() error {
			if seg.IsLazyLoad() {
				ctx, cancel := withLazyLoadTimeoutContext(ctx)
				defer cancel()
				_, err := mgr.DiskCache.Do(ctx, seg.ID(), searcher)
				if err != nil {
					log.Warn("failed to do search for disk cache", zap.Int64("segID", seg.ID()), zap.Error(err))
				}
				return err
			}
			return searcher(ctx, seg)
		})
	}
	err := g.Wait()
	DeleteSearchResults(searchResultsToClear)
	return err
}

// search will search on the historical segments the target segments in historical.
// if segIDs is not specified, it will search on all the historical segments speficied by partIDs.
// if segIDs is specified, it will only search on the segments specified by the segIDs.
// if partIDs is empty, it means all the partitions of the loaded collection or all the partitions loaded.
func SearchHistorical(ctx context.Context, manager *Manager, searchReq *SearchRequest, collID int64, partIDs []int64, segIDs []int64) ([]*SearchResult, []Segment, error) {
	if ctx.Err() != nil {
		return nil, nil, ctx.Err()
	}

	segments, err := validateOnHistorical(ctx, manager, collID, partIDs, segIDs)
	if err != nil {
		return nil, nil, err
	}
	searchResults, err := searchSegments(ctx, manager, segments, SegmentTypeSealed, searchReq)
	return searchResults, segments, err
}

// searchStreaming will search all the target segments in streaming
// if partIDs is empty, it means all the partitions of the loaded collection or all the partitions loaded.
func SearchStreaming(ctx context.Context, manager *Manager, searchReq *SearchRequest, collID int64, partIDs []int64, segIDs []int64) ([]*SearchResult, []Segment, error) {
	if ctx.Err() != nil {
		return nil, nil, ctx.Err()
	}

	segments, err := validateOnStream(ctx, manager, collID, partIDs, segIDs)
	if err != nil {
		return nil, nil, err
	}
	searchResults, err := searchSegments(ctx, manager, segments, SegmentTypeGrowing, searchReq)
	return searchResults, segments, err
}

func SearchHistoricalStreamly(ctx context.Context, manager *Manager, searchReq *SearchRequest,
	collID int64, partIDs []int64, segIDs []int64, streamReduce func(result *SearchResult) error,
) ([]Segment, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	segments, err := validateOnHistorical(ctx, manager, collID, partIDs, segIDs)
	if err != nil {
		return segments, err
	}
	err = searchSegmentsStreamly(ctx, manager, segments, searchReq, streamReduce)
	if err != nil {
		return segments, err
	}
	return segments, nil
}
