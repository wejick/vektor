package main

import (
	"fmt"

	v "github.com/wejick/vektor/hnsw"
)

func main() {
	graph := v.NewHNSW(
		v.HNSWOption{
			M:              5,
			EfConstruction: 200,
			EfSearch:       20,
			MaxLevel:       3,
			VectorDim:      2,
			Size:           1000,

			RNG: &v.StaticRNGMachine{Value: 0.03},
		})

	graph.AddVector([]float32{1, 1})
	graph.AddVector([]float32{1, 2})
	graph.AddVector([]float32{1, 3})
	graph.AddVector([]float32{0, 1})
	graph.AddVector([]float32{2, 1})
	graph.AddVector([]float32{3, 1})
	graph.AddVector([]float32{4, 1})
	graph.AddVector([]float32{5, 1})
	graph.AddVector([]float32{6, 1})
	graph.AddVector([]float32{7, 8})
	graph.AddVector([]float32{8, 9})
	graph.AddVector([]float32{17, 18})
	graph.AddVector([]float32{21, 22})
	graph.AddVector([]float32{25, 26})
	graph.AddVector([]float32{30, 31})
	graph.AddVector([]float32{35, 36})

	graph.PrintGraph()

	fmt.Println(graph.Search([]float32{17, 18}, 5))
	fmt.Println(graph.Search([]float32{17, 18}, 5))
}
