package main

import (
	v "github.com/wejick/vektor/hnsw"
)

func main() {
	graph := v.NewHNSW(
		v.HNSWOption{
			M:              3,
			EfConstruction: 200,
			EfSearch:       20,
			MaxLevel:       5,
			VectorDim:      4,
			Size:           1000,
		})

	graph.AddVector([]float64{1, 2, 3, 4})
	graph.AddVector([]float64{5, 6, 7, 8})
	graph.AddVector([]float64{9, 10, 11, 12})
	graph.AddVector([]float64{1, 2, 3, 2})
	graph.AddVector([]float64{17, 18, 19, 20})
	graph.AddVector([]float64{21, 22, 23, 24})
	graph.AddVector([]float64{25, 26, 27, 28})

	graph.PrintGraph()
}
