package hnsw

import (
	"fmt"
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

type distanceComputer func(vec1, vec2 []float64) float64

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

	distanceComputerFunc distanceComputer

	EfConstruction int
	EfSearch       int

	mL  float64 // mL = 1 / log(M)
	rng *rand.Rand

	vectors [][]float64 // Vectors in the graph
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
		M:                    option.M,
		EfConstruction:       option.EfConstruction,
		EfSearch:             option.EfSearch,
		MaxLevel:             option.MaxLevel,
		vectorDim:            option.VectorDim,
		distanceComputerFunc: L2Distance, // now we only have this. hardcode
		curMaxLevel:          0,
		rng:                  rng,
		mL:                   mL,
		vectors:              make([][]float64, 0, option.Size),
		nodes:                make([]*Node, 0, option.Size),
	}
}

func (h *HNSW) AddVector(vector []float64) (id int, err error) {

	// can't add if dimension is different
	if len(vector) != h.vectorDim {
		err = fmt.Errorf("AddVector : Different vector dimension. Got %d expected %d", len(vector), h.vectorDim)
		return 0, err
	}

	maxLevel := h.genRandomMaxLevel()

	newNode := &Node{
		maxLevel: maxLevel,
	}

	h.writeLock.Lock()
	defer h.writeLock.Unlock()

	newNode.ID = len(h.nodes)
	h.vectors = append(h.vectors, vector)
	h.nodes = append(h.nodes, newNode)

	// initialize the neighbors array
	for i := 0; i <= maxLevel; i++ {
		newNode.perLayerNeighbors = append(newNode.perLayerNeighbors, []int{})
	}

	// for first node, set entry point
	if len(h.nodes) == 1 {
		h.entryPoint = newNode.ID
		h.curMaxLevel = maxLevel

		return newNode.ID, nil
	}

	// search top level
	var candidateNodeID = []int{h.entryPoint}
	var candidateDistance = []float64{0}

	// search next level until 0
	for l := h.curMaxLevel - 1; l >= 0; l-- {
		// when level higher than nodeMaxlevel, topK is 1
		if l > maxLevel {
			candidateNodeID, candidateDistance = h.searchLevel(vector, candidateNodeID, candidateDistance, l, 1)
		} else {
			candidateNodeID, candidateDistance = h.searchLevel(vector, candidateNodeID, candidateDistance, l, h.EfConstruction)
			// add candidate as neighboor on this level
			newNode.perLayerNeighbors[l] = append(newNode.perLayerNeighbors[l], candidateNodeID...)
		}
	}

	// set current max level of graph to the just added node if higher and set new entry point
	if h.curMaxLevel < maxLevel {
		h.curMaxLevel = maxLevel
		h.entryPoint = newNode.ID
	}

	return
}

// genRandomMaxLevel generate random max level. formula l = floor(-log(uniform(0,1)) * mL)
func (h *HNSW) genRandomMaxLevel() int {
	uniform := h.rng.Float64()

	return int(math.Floor(-math.Log(uniform) * h.mL))
}

func (h *HNSW) searchLevelInternal(vectorToSearch []float64, entrypointNode []int, distanceToEntrypoint []float64, level int) (result priorityQueueMin) {
	if len(entrypointNode) != len(distanceToEntrypoint) {
		return
	}

	visited := make(map[int]bool)
	candidate := newPriorityQueueMax(h.EfConstruction)
	result = newPriorityQueueMin(h.EfConstruction)

	var farthestResult float64

	for idx := 0; idx < len(entrypointNode); idx++ {
		candidate.Push(&pqItem{Value: entrypointNode[idx], Priority: distanceToEntrypoint[idx]})
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
			toVisit.Priority = h.distanceComputerFunc(vectorToSearch, h.vectors[toVisit.Value])
		}

		// update farthest result
		if toVisit.Priority > farthestResult {
			farthestResult = toVisit.Priority
		}

		// stop criteria : if the shortest candidate farther than the farthestResult, stop early.
		if toVisit.Priority > farthestResult {
			break
		}

		// add to result
		result.Push(toVisit)
		visited[toVisit.Value] = true

		// add neighboor as candidate
		for _, nodeID := range h.nodes[toVisit.Value].perLayerNeighbors[level] {
			candidate.Push(&pqItem{Value: nodeID})
		}
	}

	return
}

// searchLevel search within defined level
func (h *HNSW) searchLevel(vectorToSearch []float64, entrypointNode []int, distanceToEntrypoint []float64, level int, topK int) (resultNodeID []int, resultDistance []float64) {
	if len(entrypointNode) != len(distanceToEntrypoint) {
		return
	}

	result := h.searchLevelInternal(vectorToSearch, entrypointNode, distanceToEntrypoint, level)

	// put topK nearest as result
	for k := 0; k < topK; k++ {
		if result.Len() == 0 {
			break
		}
		res := result.Pop().(*pqItem)
		resultNodeID = append(resultNodeID, res.Value)
		resultDistance = append(resultDistance, res.Priority)
	}

	return
}

// Function to pretty print the HNSW graph
func (h *HNSW) PrintGraph() {
	if h == nil || len(h.nodes) == 0 {
		fmt.Println("Graph is empty or not initialized.")
		return
	}

	fmt.Printf("HNSW Graph (Current Max Level: %d, Entry Point: %d)\n", h.curMaxLevel, h.entryPoint)
	fmt.Println("===================================================")

	// Iterate from the highest current level down to level 0
	for level := h.curMaxLevel; level >= 0; level-- {
		fmt.Printf("\n--- Level %d ---\n", level)
		foundNodesAtLevel := false

		// Iterate through all nodes in the graph
		for _, node := range h.nodes {
			if node == nil {
				continue // Skip nil nodes if any (should ideally not happen in a well-managed graph)
			}

			// Check if the node exists at the current level
			if node.maxLevel >= level {
				foundNodesAtLevel = true
				fmt.Printf("  Node %d: ", node.ID)

				// Check if the perLayerNeighbors slice is large enough for this level
				if level < len(node.perLayerNeighbors) {
					neighborsAtLevel := node.perLayerNeighbors[level]
					if len(neighborsAtLevel) > 0 {
						fmt.Print("Neighbors: [")
						for i, neighborID := range neighborsAtLevel {
							fmt.Printf("%d", neighborID)
							if i < len(neighborsAtLevel)-1 {
								fmt.Print(", ")
							}
						}
						fmt.Println("]")
					} else {
						fmt.Println("Neighbors: []")
					}
				} else {
					// This case implies the node.perLayerNeighbors was not fully populated
					// up to node.maxLevel for this specific 'level'.
					// This might indicate an issue in graph construction or that the node
					// has no connections defined at this level despite existing up to its maxLevel.
					fmt.Println("Neighbors: (no connections defined at this level in perLayerNeighbors)")
				}
			}
		}
		if !foundNodesAtLevel {
			fmt.Println("  (No nodes at this level)")
		}
	}
	fmt.Println("\n===================================================")
}
