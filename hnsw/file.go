package hnsw

import (
	"encoding/json"
	"io/ioutil"
	"math/rand"
	"os"
	"time"
)

type HNSWOnDisk struct {
	M               int
	MaxLevel        int
	VectorDim       int
	Size            int
	NormalizeVector bool

	CurMaxLevel int
	EntryPoint  int

	EfConstruction int
	EfSearch       int

	ML                   float64 // mL = 1 / log(M)
	RNGMachine           string
	DistanceComputerFunc string

	Vectors [][]float32 // Vectors in the graph
	Nodes   []*Node     // Nodes in the graph
}

func LoadFromDisk(filepath string) (*HNSW, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	onDisk := &HNSWOnDisk{}
	var jsonOnDisk []byte
	jsonOnDisk, err = ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(jsonOnDisk, onDisk)
	if err != nil {
		return nil, err
	}

	source := rand.NewSource(time.Now().UnixNano())
	rng := RNGMachine(rand.New(source))

	index := &HNSW{
		M:               onDisk.M,
		EfConstruction:  onDisk.EfConstruction,
		EfSearch:        onDisk.EfSearch,
		size:            onDisk.Size,
		normalizeVector: onDisk.NormalizeVector,
		MaxLevel:        onDisk.MaxLevel,
		vectorDim:       onDisk.VectorDim,
		curMaxLevel:     onDisk.CurMaxLevel,
		entryPoint:      onDisk.EntryPoint,
		rng:             rng,
		mL:              onDisk.ML,
		vectors:         onDisk.Vectors,
		nodes:           onDisk.Nodes,
	}

	switch onDisk.DistanceComputerFunc {
	case l2DistanceName:
		index.distanceComputerFunc = &L2Distance{}
	case L2SquaredDistanceName:
		index.distanceComputerFunc = &L2SquaredDistance{}
	default:
		index.distanceComputerFunc = &L2SquaredDistance{}
	}

	return index, nil
}
