package hnsw

type HNSWOnDisk struct {
	M         int
	MaxLevel  int
	VectorDim int
	Size      int

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
