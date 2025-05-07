package hnsw

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

const defaultM = 16
const defaultEfConstruction = 200
const defaultEfSearch = 20
const defaultMaxLevel = 16
const defaultVectorDim = 128

const defaultSize = 1000

type distanceComputer func(vectorSrc, vectorDst []float32) float32

type HNSWOption struct {
	M              int
	EfConstruction int
	EfSearch       int
	MaxLevel       int
	VectorDim      int

	// graph size, this is not hard limit as Go will grow the slice
	// but it is a good idea to set it to a reasonable value
	// to avoid unnecessary memory allocation & copying
	Size int64
}

type HNSW struct {
	M           int
	MaxLevel    int
	vectorDim   int
	curMaxLevel int
	entryPoint  int

	EfConstruction int
	EfSearch       int

	mL  float64 // mL = 1 / log(M)
	rng *rand.Rand

	vectors [][]float32 // Vectors in the graph
	nodes   []*Node     // Nodes in the graph

	writeLock sync.Mutex
	curMaxID  int // Current max ID
}

type Node struct {
	ID                int
	perLayerNeighbors [][]int // Neighbors per layer
	maxLevel          int
}

// NewHNSW creates a new HNSW graph with the given options
func NewHNSW(option HNSWOption) *HNSW {
	if option.M == 0 {
		option.M = defaultM
	}
	if option.EfConstruction == 0 {
		option.EfConstruction = defaultEfConstruction
	}
	if option.EfSearch == 0 {
		option.EfSearch = defaultEfSearch
	}
	if option.MaxLevel == 0 {
		option.MaxLevel = defaultMaxLevel
	}
	if option.VectorDim == 0 {
		option.VectorDim = defaultVectorDim
	}

	if option.Size == 0 {
		option.Size = defaultSize
	}

	// Seed the random number generator ONCE
	source := rand.NewSource(time.Now().UnixNano())
	rng := rand.New(source)

	// Pre-calculate mL
	mL := 1.0 / math.Log(float64(option.M))

	return &HNSW{
		M:              option.M,
		EfConstruction: option.EfConstruction,
		EfSearch:       option.EfSearch,
		MaxLevel:       option.MaxLevel,
		vectorDim:      option.VectorDim,
		curMaxLevel:    0,
		rng:            rng,
		mL:             mL,
		vectors:        make([][]float32, 0, option.Size),
		nodes:          make([]*Node, 0, option.Size),
	}
}

func (h *HNSW) AddVector(vector []float32) (id int64) {

	maxLevel := h.genRandomMaxLevel()

	newNode := &Node{
		maxLevel:          maxLevel,
		perLayerNeighbors: make([][]int, maxLevel),
	}

	h.writeLock.Lock()
	defer h.writeLock.Unlock()

	newNode.ID = len(h.nodes)
	h.vectors = append(h.vectors, vector)
	h.nodes = append(h.nodes, newNode)

	// search top level
	candidateNodeID, candidateDistance := h.searchLevel(vector, []int{h.entryPoint}, []float32{0}, h.curMaxLevel, 1, nil)

	// search next level until 0
	for l := h.curMaxLevel - 1; l >= 0; l-- {
		// when level higher than nodeMaxlevel, topK is 1
		if l > maxLevel {
			candidateNodeID, candidateDistance = h.searchLevel(vector, candidateNodeID, candidateDistance, l, 1, nil)
		} else {
			candidateNodeID, candidateDistance = h.searchLevel(vector, candidateNodeID, candidateDistance, l, h.EfConstruction, nil)
			// add candidate as neighboor on this level
			newNode.perLayerNeighbors[l] = append(newNode.perLayerNeighbors[l], candidateNodeID...)
		}
	}

	// set current max level of graph to the just added node if higher
	if h.curMaxLevel < maxLevel {
		h.curMaxLevel = maxLevel
	}

	return
}

// genRandomMaxLevel generate random max level. formula l = floor(-log(uniform(0,1)) * mL)
func (h *HNSW) genRandomMaxLevel() int {
	uniform := h.rng.Float64()

	return int(math.Floor(-math.Log(uniform) * h.mL))
}

// searchLevel search within defined level
func (h *HNSW) searchLevel(vectorToSearch []float32, entrypointNode []int, distanceToEntrypoint []float32, level int, topK int, distanceFunc distanceComputer) (resultNodeID []int, resultDistance []float32) {
	if len(entrypointNode) != len(distanceToEntrypoint) {
		return
	}

	resultNodeID = make([]int, 0, topK)

	visited := make(map[int]bool)
	candidate := newPriorityQueueMax(h.EfConstruction)
	result := newPriorityQueueMin(h.EfConstruction)

	var farthestResult float32

	for idx := 0; idx < len(entrypointNode); idx++ {
		candidate.Push(pqItem{Value: entrypointNode[idx], Priority: distanceToEntrypoint[idx]})
	}

	for candidate.Len() > 0 {
		toVisit := candidate.Pop().(*pqItem)

		// skip visited node
		if visited[toVisit.Value] {
			continue
		}

		// calculate the distance
		// Distance is being used for the priority queue
		if toVisit.Priority == 0 {
			toVisit.Priority = distanceFunc(vectorToSearch, h.vectors[toVisit.Value])
		}

		// update farthest result
		if toVisit.Priority > farthestResult {
			farthestResult = toVisit.Priority
		}

		// stop criteria : if the shortest candidate farther than the farther result, stop early.
		if result.Len() >= topK && toVisit.Priority > farthestResult {
			break
		}

		// add to result
		result.Push(toVisit)
		visited[toVisit.Value] = true

		// add neighboor as candidate
		for _, nodeID := range h.nodes[toVisit.Value].perLayerNeighbors[level] {
			candidate.Push(pqItem{Value: nodeID})
		}
	}

	// put topK nearest as result
	for k := 0; k < topK; k++ {
		res := result.Pop().(*pqItem)
		resultNodeID = append(resultNodeID, res.Value)
		resultDistance = append(resultDistance, res.Priority)
	}

	return
}
