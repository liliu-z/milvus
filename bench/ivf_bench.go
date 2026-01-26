package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	collectionName = "cohere_medium_1m"
	dim            = 768
	topK           = 100
	numQueries     = 1000
	numNeighbors   = 1000
	searchDuration = 2 * time.Minute
	dataPath       = "/home/ubuntu/data/cohere_medium_1m/"
)

var concurrencyLevels = []int{1, 5, 10, 20, 30}
var nprobeValues = []int{16, 32, 64, 128}

func loadTestVectors(path string) ([][]float32, error) {
	data, err := os.ReadFile(path + "test_vectors.bin")
	if err != nil {
		return nil, err
	}
	numVecs := len(data) / (dim * 4)
	vectors := make([][]float32, numVecs)
	for i := 0; i < numVecs; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			offset := (i*dim + j) * 4
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			vec[j] = math.Float32frombits(bits)
		}
		vectors[i] = vec
	}
	return vectors, nil
}

func loadGroundTruth(path string) ([][]int64, error) {
	data, err := os.ReadFile(path + "neighbors.bin")
	if err != nil {
		return nil, err
	}
	numRows := len(data) / (numNeighbors * 8)
	gt := make([][]int64, numRows)
	for i := 0; i < numRows; i++ {
		row := make([]int64, numNeighbors)
		for j := 0; j < numNeighbors; j++ {
			offset := (i*numNeighbors + j) * 8
			row[j] = int64(binary.LittleEndian.Uint64(data[offset : offset+8]))
		}
		gt[i] = row
	}
	return gt, nil
}

func computeRecall(results []int64, gt []int64, k int) float64 {
	gtSet := make(map[int64]struct{}, k)
	for i := 0; i < k && i < len(gt); i++ {
		gtSet[gt[i]] = struct{}{}
	}
	hits := 0
	for _, id := range results {
		if _, ok := gtSet[id]; ok {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

type SearchResult struct {
	latency time.Duration
	recall  float64
}

func runIVFBenchmark(ctx context.Context, c client.Client, vectors [][]float32, gt [][]int64, nprobe int) {
	fmt.Printf("\n=== IVF_FLAT nprobe=%d ===\n", nprobe)

	sp, _ := entity.NewIndexIvfFlatSearchParam(nprobe)

	for _, concurrency := range concurrencyLevels {
		var (
			totalOps   int64
			allResults []SearchResult
			mu         sync.Mutex
			wg         sync.WaitGroup
		)

		deadline := time.Now().Add(searchDuration)
		startTime := time.Now()

		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				queryIdx := workerID % numQueries
				for time.Now().Before(deadline) {
					vec := vectors[queryIdx]
					vecCol := entity.FloatVector(vec)

					start := time.Now()
					result, err := c.Search(ctx, collectionName, nil, "", []string{"id"}, []entity.Vector{vecCol},
						"emb", entity.IP, topK, sp)
					elapsed := time.Since(start)

					if err != nil {
						fmt.Printf("Search error: %v\n", err)
						queryIdx = (queryIdx + 1) % numQueries
						continue
					}

					var resultIDs []int64
					if len(result) > 0 {
						for i := 0; i < result[0].ResultCount; i++ {
							resultIDs = append(resultIDs, result[0].IDs.(*entity.ColumnInt64).Data()[i])
						}
					}

					recall := computeRecall(resultIDs, gt[queryIdx], topK)

					mu.Lock()
					allResults = append(allResults, SearchResult{latency: elapsed, recall: recall})
					mu.Unlock()
					atomic.AddInt64(&totalOps, 1)

					queryIdx = (queryIdx + concurrency) % numQueries
				}
			}(i)
		}

		wg.Wait()
		totalTime := time.Since(startTime)

		if len(allResults) == 0 {
			fmt.Println("No results collected!")
			continue
		}

		latencies := make([]float64, len(allResults))
		var totalRecall float64
		for i, r := range allResults {
			latencies[i] = float64(r.latency.Microseconds())
			totalRecall += r.recall
		}

		sort.Float64s(latencies)

		_ = average(latencies)
		p50 := percentile(latencies, 50)
		p99 := percentile(latencies, 99)
		qps := float64(totalOps) / totalTime.Seconds()
		avgRecall := totalRecall / float64(len(allResults))

		fmt.Printf("C=%d: QPS=%.0f, P50=%.0fus, P99=%.0fus, Recall=%.3f\n",
			concurrency, qps, p50, p99, avgRecall)
	}
}

func average(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * p / 100.0)
	return sorted[idx]
}

func main() {
	fmt.Println("Loading test vectors...")
	vectors, err := loadTestVectors(dataPath)
	if err != nil {
		fmt.Printf("Failed to load vectors: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d test vectors (dim=%d)\n", len(vectors), dim)

	fmt.Println("Loading ground truth...")
	gt, err := loadGroundTruth(dataPath)
	if err != nil {
		fmt.Printf("Failed to load ground truth: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d ground truth rows\n", len(gt))

	ctx := context.Background()

	fmt.Println("Connecting to Milvus...")
	c, err := client.NewClient(ctx, client.Config{
		Address: "localhost:19530",
	})
	if err != nil {
		fmt.Printf("Failed to connect: %v\n", err)
		os.Exit(1)
	}
	defer c.Close()

	fmt.Println("Loading collection...")
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		fmt.Printf("Failed to load collection: %v\n", err)
		os.Exit(1)
	}

	// Warmup
	fmt.Println("Warming up...")
	sp, _ := entity.NewIndexIvfFlatSearchParam(32)
	for i := 0; i < 10; i++ {
		vec := entity.FloatVector(vectors[i])
		_, err := c.Search(ctx, collectionName, nil, "", []string{"id"}, []entity.Vector{vec},
			"emb", entity.IP, topK, sp)
		if err != nil {
			fmt.Printf("Warmup error: %v\n", err)
		}
	}

	fmt.Println("\n========== IVF_FLAT Benchmark ==========")
	for _, nprobe := range nprobeValues {
		runIVFBenchmark(ctx, c, vectors, gt, nprobe)
	}
	fmt.Println("\n========== Benchmark Complete ==========")
}
