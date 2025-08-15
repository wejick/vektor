package main

import (
	"fmt"
	"os"
	"time"

	"github.com/kshard/fvecs"
	"github.com/wejick/vektor/hnsw"
)

var baseVector, queryVector [][]float32
var groundTruth [][]uint32

const dimension = 128

func main() {
	baseData, err := os.Open("./siftsmall/siftsmall_base.fvecs")
	if err != nil {
		panic(err)
	}
	loaderBaseData := fvecs.NewDecoder[float32](baseData)
	for {
		v, err := loaderBaseData.Read()
		if err != nil {
			break
		}
		baseVector = append(baseVector, v)
	}
	fmt.Println(len(baseVector))

	queryData, err := os.Open("./siftsmall/siftsmall_query.fvecs")
	if err != nil {
		panic(err)
	}
	loaderQueryData := fvecs.NewDecoder[float32](queryData)
	for {
		v, err := loaderQueryData.Read()
		if err != nil {
			break
		}
		queryVector = append(queryVector, v)
	}
	fmt.Println(len(queryVector))

	groundTruthData, err := os.Open("./siftsmall/siftsmall_groundtruth.ivecs")
	if err != nil {
		panic(err)
	}
	loaderGroundTruthData := fvecs.NewDecoder[uint32](groundTruthData)
	for {
		v, err := loaderGroundTruthData.Read()
		if err != nil {
			break
		}
		groundTruth = append(groundTruth, v)
	}
	fmt.Println(len(groundTruth))

	hnswIndex := hnsw.NewHNSW(hnsw.HNSWOption{
		M:              16,
		EfConstruction: 256,
		EfSearch:       256,
		MaxLevel:       5,
		VectorDim:      dimension,
		Size:           len(baseVector),
	})

	now := time.Now()
	indexBase(baseVector, hnswIndex)
	fmt.Printf("Indexing took %v\n", time.Since(now))

	now = time.Now()
	result := searchQuery(queryVector, hnswIndex, 100)
	fmt.Printf("Searching took %v\n", time.Since(now))

	recall := calculateRecall(result, groundTruth)
	fmt.Println("Recall:", recall)
}

func indexBase(vectorData [][]float32, index *hnsw.HNSW) {
	for _, v := range vectorData {
		index.AddVector(v)
	}
}

func searchQuery(vectorData [][]float32, index *hnsw.HNSW, topK int) (results map[int][]int) {
	results = make(map[int][]int)
	for i, v := range vectorData {
		result, _, err := index.Search(v, topK)
		if err != nil {
			panic(err)
		}
		results[i] = result
	}
	return results
}

func calculateRecall(results map[int][]int, groundTruth [][]uint32) float64 {
	// Ensure we don't divide by zero if there are no results.
	if len(results) == 0 {
		return 0.0
	}

	var totalRecall float64

	// Iterate through each query's results.
	for queryIndex, retrievedIndices := range results {
		// Ensure the ground truth for this query exists.
		if queryIndex >= len(groundTruth) {
			continue
		}

		groundTruthForQuery := groundTruth[queryIndex]

		// Put the ground truth into a set for efficient lookup.
		groundTruthSet := make(map[uint32]struct{}, len(groundTruthForQuery))
		for _, id := range groundTruthForQuery {
			groundTruthSet[id] = struct{}{}
		}

		// Count how many of the retrieved items are in the ground truth set.
		var intersectionCount int
		for _, retrievedID := range retrievedIndices {
			if _, found := groundTruthSet[uint32(retrievedID)]; found {
				intersectionCount++
			}
		}

		// Calculate the recall for this specific query.
		// Recall = (Number of relevant items retrieved) / (Total number of relevant items)
		var queryRecall float64
		if len(groundTruthForQuery) > 0 {
			queryRecall = float64(intersectionCount) / float64(len(groundTruthForQuery))
		}

		// Add the query's recall to the total.
		totalRecall += queryRecall
	}

	// Return the average recall across all queries.
	return totalRecall / float64(len(results))
}
