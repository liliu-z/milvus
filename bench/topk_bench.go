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
	numQueries     = 1000
	numNeighbors   = 1000
	searchDuration = 1 * time.Minute
	dataPath       = "/home/ubuntu/data/cohere_medium_1m/"
	concurrency    = 10
)

var topKValues = []int{10, 50, 100, 200}

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

func runTopKBenchmark(ctx context.Context, c client.Client, vectors [][]float32, gt [][]int64, topK int, ef int) {
	fmt.Printf("\n=== topK=%d, ef=%d, C=%d ===\n", topK, ef, concurrency)

	sp, _ := entity.NewIndexHNSWSearchParam(ef)

	var (
		totalOps   int64
		latencies  []float64
		recalls    []float64
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
				latencies = append(latencies, float64(elapsed.Microseconds()))
				recalls = append(recalls, recall)
				mu.Unlock()
				atomic.AddInt64(&totalOps, 1)

				queryIdx = (queryIdx + concurrency) % numQueries
			}
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	sort.Float64s(latencies)

	p50 := latencies[len(latencies)/2]
	p99 := latencies[int(float64(len(latencies))*0.99)]
	qps := float64(totalOps) / totalTime.Seconds()

	var totalRecall float64
	for _, r := range recalls {
		totalRecall += r
	}
	avgRecall := totalRecall / float64(len(recalls))

	fmt.Printf("QPS=%.0f, P50=%.0fus, P99=%.0fus, Recall=%.3f\n", qps, p50, p99, avgRecall)
}

func main() {
	fmt.Println("Loading test vectors...")
	vectors, err := loadTestVectors(dataPath)
	if err != nil {
		fmt.Printf("Failed to load vectors: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Loading ground truth...")
	gt, err := loadGroundTruth(dataPath)
	if err != nil {
		fmt.Printf("Failed to load ground truth: %v\n", err)
		os.Exit(1)
	}

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
	c.LoadCollection(ctx, collectionName, false)

	fmt.Println("\n========== TopK Benchmark ==========")

	// Test topK with ef = max(topK, 100)
	for _, topK := range topKValues {
		ef := topK
		if ef < 100 {
			ef = 100
		}
		runTopKBenchmark(ctx, c, vectors, gt, topK, ef)
	}

	fmt.Println("\n========== Benchmark Complete ==========")
}
