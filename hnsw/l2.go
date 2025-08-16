package hnsw

import (
	"math"
)

type distanceComputer interface {
	CalcDistance(vec1, vec2 []float32) float32
	GetName() string
}

const l2DistanceName = "L2Distance"
const L2SquaredDistanceName = "L2SquaredDistance"

type (
	L2Distance        struct{}
	L2SquaredDistance struct{}
)

// L2Distance calculates the Euclidean (L2) distance between two vectors.
// It returns an error if the vectors have different lengths.
func (L2 *L2Distance) CalcDistance(vec1, vec2 []float32) float32 {
	var sumOfSquares float32
	for i := 0; i < len(vec1); i++ {
		diff := vec1[i] - vec2[i]
		sumOfSquares += diff * diff
	}

	return float32(math.Sqrt(float64(sumOfSquares)))
}

func (L2 *L2Distance) GetName() string {
	return l2DistanceName
}

func (L2 *L2SquaredDistance) CalcDistance(vec1, vec2 []float32) (sumOfSquares float32) {
	var diff float32
	for i := range vec1 {
		diff = vec1[i] - vec2[i]
		sumOfSquares += diff * diff
	}
	return sumOfSquares
}

func (L2 *L2SquaredDistance) GetName() string {
	return L2SquaredDistanceName
}
