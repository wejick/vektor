package hnsw

type RNGMachine interface {
	Float64() float64
}

type StaticRNGMachine struct {
	Value float64
}

func (s *StaticRNGMachine) Float64() float64 {
	return s.Value
}
