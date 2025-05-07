package hnsw

import (
	"container/heap"
)

// An pqItem is something we manage in a priority queue.
type pqItem struct {
	Value    int     // The value of the item; arbitrary.
	Priority float32 // The priority of the item in the queue.
	// The index is needed by update and is maintained by the heap.Interface methods.
	index int // The index of the item in the heap.
}

// A priorityQueue implements heap.Interface and holds Items.
type priorityQueueMax []*pqItem
type priorityQueueMin []*pqItem

func (pq priorityQueueMax) Len() int { return len(pq) }
func (pq priorityQueueMin) Len() int { return len(pq) }

func (pq priorityQueueMax) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].Priority > pq[j].Priority
}

func (pq priorityQueueMin) Less(i, j int) bool {
	// We want Pop to give us the lowest
	return pq[i].Priority < pq[j].Priority
}

func (pq priorityQueueMax) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq priorityQueueMin) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueueMax) Push(x any) {
	n := len(*pq)
	item := x.(*pqItem)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *priorityQueueMin) Push(x any) {
	n := len(*pq)
	item := x.(*pqItem)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *priorityQueueMax) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // don't stop the GC from reclaiming the item eventually
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

func (pq *priorityQueueMin) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // don't stop the GC from reclaiming the item eventually
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

// update modifies the priority and value of an Item in the queue.
// func (pq *priorityQueueMax) update(item *pqItem, value int, priority float32) {
// 	item.Value = value
// 	item.Priority = priority
// 	heap.Fix(pq, item.index)
// }

// update modifies the priority and value of an Item in the queue.
// func (pq *priorityQueueMin) update(item *pqItem, value int, priority float32) {
// 	item.Value = value
// 	item.Priority = priority
// 	heap.Fix(pq, item.index)
// }

func newPriorityQueueMax(size int) priorityQueueMax {
	pq := make(priorityQueueMax, 0, size)

	heap.Init(&pq)

	return pq
}

func newPriorityQueueMin(size int) priorityQueueMin {
	pq := make(priorityQueueMin, 0, size)

	heap.Init(&pq)

	return pq
}
