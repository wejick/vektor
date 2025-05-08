# vektor

learning hnsw in go

```
HNSW Graph (Current Max Level: 3, Entry Point: 12)
===================================================

--- Level 3 ---
  Node 12: Neighbors: []

--- Level 2 ---
  Node 12: Neighbors: []

--- Level 1 ---
  Node 6: Neighbors: []
  Node 7: Neighbors: []
  Node 11: Neighbors: []
  Node 12: Neighbors: []

--- Level 0 ---
  Node 0: Neighbors: []
  Node 1: Neighbors: []
  Node 2: Neighbors: []
  Node 3: Neighbors: []
  Node 4: Neighbors: []
  Node 5: Neighbors: []
  Node 6: Neighbors: [7, 8, 9, 10, 11]
  Node 7: Neighbors: [6, 8, 9, 10, 11]
  Node 8: Neighbors: [7, 6, 9, 10, 11]
  Node 9: Neighbors: [7, 8, 6, 10, 11]
  Node 10: Neighbors: [7, 8, 9, 6, 11]
  Node 11: Neighbors: [7, 8, 9, 10, 6]
  Node 12: Neighbors: [11, 10, 9, 8, 7]
  Node 13: Neighbors: [10, 9, 8, 6, 7]
  Node 14: Neighbors: [8, 9, 10, 6, 11]
  Node 15: Neighbors: [10, 9, 8, 6, 7]

===================================================
[8 9 10 6 11] [20.248456731316587 14.142135623730951 12.727922061357855 21.400934559032695 0] <nil>
[8 9 10 6 11] [20.248456731316587 14.142135623730951 12.727922061357855 21.400934559032695 0] <nil>
```