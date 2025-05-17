package hnsw

import (
	"container/heap"
	"testing"
)

func TestPriorityQueueMax_Basic(t *testing.T) {
	pq := newPriorityQueueMax(0)

	heap.Push(&pq, &pqItem{Value: 1, Priority: 2.0})
	heap.Push(&pq, &pqItem{Value: 2, Priority: 5.0})
	heap.Push(&pq, &pqItem{Value: 3, Priority: 1.0})
	heap.Push(&pq, &pqItem{Value: 4, Priority: 3.0})

	// Should pop in order of highest priority first
	expectedOrder := []int{2, 4, 1, 3}
	for i, expected := range expectedOrder {
		item := heap.Pop(&pq).(*pqItem)
		if item.Value != expected {
			t.Errorf("pop %d: expected value %d, got %d", i, expected, item.Value)
		}
	}

	if pq.Len() != 0 {
		t.Errorf("expected empty queue after pops, got len %d", pq.Len())
	}
}

func TestPriorityQueueMin_Basic(t *testing.T) {
	pq := newPriorityQueueMin(0)

	heap.Push(&pq, &pqItem{Value: 3, Priority: 3.0})
	heap.Push(&pq, &pqItem{Value: 4, Priority: 4.0})
	heap.Push(&pq, &pqItem{Value: 1, Priority: 1.0})
	heap.Push(&pq, &pqItem{Value: 2, Priority: 2.0})

	// Should pop in order of lowest priority first
	expectedOrder := []int{1, 2, 3, 4}
	for i, expected := range expectedOrder {
		item := heap.Pop(&pq).(*pqItem)
		if item.Value != expected {
			t.Errorf("pop %d: expected value %d, got %d", i, expected, item.Value)
		}
	}

	if pq.Len() != 0 {
		t.Errorf("expected empty queue after pops, got len %d", pq.Len())
	}
}
