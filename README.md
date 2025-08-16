# vektor

learning hnsw in go

# Run recall testing
We're using [ANN_SIFT10K](http://corpus-texmex.irisa.fr/) dataset to run the test.

```shell
$cd hnsw_recall_test
$go run main.go --rebuild
10000
100
100
Rebuilding index...
Indexing took 1m39.847052777s
Searching took 2.314115739s
Recall: 0.9155000000000001
```

# Getting started
Using the library is straightforward.

```go
import v "github.com/wejick/vektor/hnsw"

func main() {
	graph := v.NewHNSW(
		v.HNSWOption{
			M:              5,
			EfConstruction: 200,
			EfSearch:       20,
			MaxLevel:       3,
			VectorDim:      2,
			Size:           1000,
		})

  graph.AddVector([]float32{1, 1}) // Adding vector

  graph.Search([]float32{17, 18}, 5) // Doing ANN Search
}
```

# Notice
1. AddVector is not threadsafe
