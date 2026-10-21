# Measured benchmark findings

- Vertex counts: 100, 500, 1000, 2000.
- Graph densities: 0.1, 0.5.
- MPI process counts: 1, 2, 4, 8, 16, including one-process MPI.
- Verified main-phase measurements: 240; 8 distinct graph hashes.
- MPI configurations slower than the matching sequential mean: 9/40.
- Highest measured speedup: 3.6429× at 2000 vertices, density 0.5, 16 process(es).
- Lowest measured speedup: 0.5355× at 100 vertices, density 0.1, 16 process(es).

Times cover complete public dijkstra() calls. MPI times are the maximum of rank-local elapsed times, collected after timing. Sequential baselines were measured separately. Pilot and discarded warm-up samples are excluded. Standard deviations are sample standard deviations (ddof=1).

## Largest measured graph: 2000 vertices

| Density | Implementation | Processes | Mean seconds | Sample SD | Speedup |
| --- | --- | --- | --- | --- | --- |
| 0.1 | sequential | 1 | 1.806245 | 0.019717 | 1.0000× |
| 0.1 | parallel | 1 | 1.880913 | 0.047751 | 0.9603× |
| 0.1 | parallel | 2 | 1.053597 | 0.022379 | 1.7144× |
| 0.1 | parallel | 4 | 0.648496 | 0.020240 | 2.7853× |
| 0.1 | parallel | 8 | 0.593253 | 0.024061 | 3.0446× |
| 0.1 | parallel | 16 | 0.513928 | 0.018611 | 3.5146× |
| 0.5 | sequential | 1 | 2.328957 | 0.030030 | 1.0000× |
| 0.5 | parallel | 1 | 2.308844 | 0.050467 | 1.0087× |
| 0.5 | parallel | 2 | 1.332887 | 0.036357 | 1.7473× |
| 0.5 | parallel | 4 | 0.823301 | 0.027180 | 2.8288× |
| 0.5 | parallel | 8 | 0.760403 | 0.035580 | 3.0628× |
| 0.5 | parallel | 16 | 0.639311 | 0.010175 | 3.6429× |

These results describe this hardware, runtime, graph set, and timing scope. The measurements alone do not establish a cause for observed slowdowns.
