package hnsw

import (
	"math"
)

// L2Distance calculates the Euclidean (L2) distance between two vectors.
// It returns an error if the vectors have different lengths.
func L2Distance(vec1, vec2 []float64) float64 {
	// Handle empty vectors (distance is 0).
	if len(vec1) == 0 {
		return 0
	}

	var sumOfSquares float64
	for i := 0; i < len(vec1); i++ {
		diff := vec1[i] - vec2[i]
		sumOfSquares += diff * diff
	}

	return math.Sqrt(sumOfSquares)
}
