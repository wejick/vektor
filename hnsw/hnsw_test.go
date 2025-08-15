package hnsw

import (
	"container/heap"
	"testing"
)

const staticRNG = 0.03

func TestHNSW_linkNeighborNode(t *testing.T) {
	tree := NewHNSW(HNSWOption{
		M:              5,
		EfConstruction: 5,
		EfSearch:       5,
		MaxLevel:       5,
		VectorDim:      2,

		RNG: &StaticRNGMachine{Value: staticRNG},
	})

	id1, _ := tree.AddVector([]float32{0, 0})
	id2, _ := tree.AddVector([]float32{1, 1})
	id3, _ := tree.AddVector([]float32{1, 2})
	id4, _ := tree.AddVector([]float32{1, 3})
	id5, _ := tree.AddVector([]float32{1, 4})
	id6, _ := tree.AddVector([]float32{1, 10})
	id7, _ := tree.AddVector([]float32{1, 6})

	// reset the M of the node 1
	tree.nodes[id1].perLevelNeighbors[0] = make([]int, 0)

	tree.linkNeighborNode(id2, id1, 0)

	if len(tree.nodes[id1].perLevelNeighbors[0]) != 1 {
		t.Errorf("expected 1 neighbor, got %d", len(tree.nodes[id1].perLevelNeighbors[0]))
	}

	tree.linkNeighborNode(id3, id1, 0)
	tree.linkNeighborNode(id4, id1, 0)
	tree.linkNeighborNode(id5, id1, 0)
	tree.linkNeighborNode(id6, id1, 0)
	tree.linkNeighborNode(id7, id1, 0)

	if len(tree.nodes[id1].perLevelNeighbors[0]) != tree.M {
		t.Errorf("expected %d neighbor, got %d", tree.M, len(tree.nodes[id1].perLevelNeighbors[0]))
	}

	// Check only nearest neighboor is linked
	// id6 is farthest
	expectedM := []int{id2, id3, id4, id5, id7}
	for idx := range tree.nodes[id1].perLevelNeighbors[0] {
		if tree.nodes[id1].perLevelNeighbors[0][idx] != expectedM[idx] {
			t.Errorf("expected %d neighbor id, got %d. idx %d", expectedM[idx], tree.nodes[id1].perLevelNeighbors[0][idx], idx)
		}
	}
}

func BenchmarkHNSW_linkNeighborNode(b *testing.B) {
	tree := NewHNSW(HNSWOption{
		M:              16,
		EfConstruction: 16,
		EfSearch:       16,
		MaxLevel:       4,
		VectorDim:      32,

		RNG: &StaticRNGMachine{Value: staticRNG},
	})

	// Add a base node
	idBase, _ := tree.AddVector(make([]float32, 32))

	// Add a pool of candidate nodes
	numCandidates := 1000
	candidateIDs := make([]int, numCandidates)
	for i := 0; i < numCandidates; i++ {
		vec := make([]float32, 32)
		for j := range vec {
			vec[j] = float32(i) + float32(j)
		}
		id, _ := tree.AddVector(vec)
		candidateIDs[i] = id
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset neighbors before each run
		tree.nodes[idBase].perLevelNeighbors[0] = make([]int, 0)
		for j := 0; j < numCandidates; j++ {
			tree.linkNeighborNode(candidateIDs[j], idBase, 0)
		}
	}
}

func TestHNSW_searchLevelInternal(t *testing.T) {
	h := NewHNSW(HNSWOption{
		M:              3,
		EfConstruction: 12,
		EfSearch:       12,
		MaxLevel:       2,
		VectorDim:      2,

		RNG: &StaticRNGMachine{Value: staticRNG},
	})

	// Add 10 vectors in a line: (0,0), (1,0), ..., (9,0)
	ids := make([]int, 10)
	for i := 0; i < 10; i++ {
		ids[i], _ = h.AddVector([]float32{float32(i), 0})
	}

	// Manually set neighbors for level 0: each node connects to previous and next (like a chain)
	for i := 0; i < 10; i++ {
		neighbors := []int{}
		if i > 0 {
			neighbors = append(neighbors, ids[i-1])
		}
		if i < 9 {
			neighbors = append(neighbors, ids[i+1])
		}
		h.nodes[ids[i]].perLevelNeighbors[0] = neighbors
	}

	// Search from node 0 at level 0
	vectorToSearch := []float32{0, 0}
	entrypointNode := []int{ids[0]}
	distanceToEntrypoint := []float32{0}
	level := 0

	result := h.searchLevelInternal(vectorToSearch, entrypointNode, distanceToEntrypoint, level)

	// Collect results
	var gotIDs []int
	for result.Len() > 0 {
		item := heap.Pop(&result).(*pqItem)
		gotIDs = append(gotIDs, item.Value)
	}

	if len(gotIDs) != 10 {
		t.Fatalf("expected 10 nodes in result, got %d", len(gotIDs))
	}

	// The IDs should be in order from 0 to 9
	for i := 0; i < 10; i++ {
		if gotIDs[i] != ids[i] {
			t.Errorf("expected id %d at index %d, got %d", ids[i], i, gotIDs[i])
		}
	}
}
