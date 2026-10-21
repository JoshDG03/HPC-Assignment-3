# HPC Assignment 3: sequential and parallel Dijkstra

Phases 1 and 2 provide native Windows / VS Code PowerShell setup, an MPI smoke
test, and sequential and parallel adjacency-matrix Dijkstra from **source vertex
0**, with correctness verification. Performance experiments, plots, and the PDF
report belong to a later phase.

The parallel implementation follows the assignment's adjacency-matrix requirement,
uppercase `comm.Scatter()` for graph distribution, and lowercase `comm.gather()`
for results. Both MPI programs follow Lecture 7's `MPI_COMM_WORLD`, rank/size,
root-0, scatter/local-sum/gather conventions (pages 4, 28–31, and 41), using
`MPI.COMM_WORLD` in mpi4py. The smoke test checks small integer buffers; parallel
Dijkstra distributes actual adjacency-matrix weights.

## Tested environment

Verified on October 20, 2026, using native Windows AMD64 and PowerShell:

| Component | Installed version |
| --- | --- |
| CPython | 3.14.3, 64-bit AMD64 |
| pip in `.venv` | 25.3 |
| NumPy | 2.5.4 |
| mpi4py | 4.1.2 |
| Intel MPI runtime (`impi-rt`) | 2021.18.1 |

No existing MPI launcher/runtime was found on PATH, in the usual Microsoft/Intel
installation locations, or in the installed-program registry entries. Intel MPI
was installed inside `.venv` using the wheel option documented in the
[official mpi4py installation instructions](https://mpi4py.readthedocs.io/en/stable/install.html#windows).
`MPI.get_vendor()` reported `('Intel MPI', (2021, 18, 1))`.

## PowerShell setup

Run from the repository root. Explicit interpreter paths mean activation and
PowerShell execution-policy changes are unnecessary. In VS Code, select
`.venv\Scripts\python.exe` with **Python: Select Interpreter**.

```powershell
Set-Location 'C:\Users\NamekianDragon\repos\HPC-Assignment-3'
py -0p
py -3.14 -c "import sys, struct; print(sys.executable); print(struct.calcsize('P') * 8)"

# Interpreter path used to create this environment:
& 'C:\Users\NamekianDragon\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

The launcher alternative for creating a new environment is
`py -3.14 -m venv .venv`. The full interpreter path above avoids the bare
`python` Windows Store alias, which failed during the initial setup.

Reuse a working Intel MPI or Microsoft MPI installation if one is already
available. `requirements.txt` deliberately contains only NumPy and mpi4py, so it
does not install a second MPI runtime. On this machine, the missing runtime was
provided locally by `impi-rt`; reproduce that choice with:

```powershell
.\.venv\Scripts\python.exe -m pip install impi-rt==2021.18.1
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
.\.venv\Scripts\python.exe -m pip freeze
```

The initial package installation actually used
`.\.venv\Scripts\python.exe -m pip install numpy mpi4py impi-rt`; the resolved
versions are recorded above and pinned for reproduction. A separate pypdf 6.21.1
reader was installed with
`.\.venv\Scripts\python.exe -m pip install --target .venv\document-tools pypdf`
solely to read the supplied PDFs; it is not an application dependency.

## MPI smoke test

Use the Intel launcher installed with this environment:

```powershell
.\.venv\Library\bin\mpiexec.exe -n 1 .\.venv\Scripts\python.exe .\mpi_smoke.py
.\.venv\Library\bin\mpiexec.exe -n 2 .\.venv\Scripts\python.exe .\mpi_smoke.py
.\.venv\Library\bin\mpiexec.exe -n 4 .\.venv\Scripts\python.exe .\mpi_smoke.py
```

For an existing system MPI runtime, use its matching `mpiexec` instead of the
project-local launcher. Always launch `.venv\Scripts\python.exe` as the worker
interpreter so every rank uses the project packages.

Root 0 creates a contiguous NumPy `int64` array, scatters two values to each rank
with uppercase `comm.Scatter()`, and collects Python dictionaries using lowercase
`comm.gather()`. Root verifies communicator sizes, rank ordering, received values,
and local sums, then prints one JSON record. Every rank receives the same exit
status; a failed check produces a nonzero exit code.

Observed results (all exit codes 0, `status: PASS`, and empty `errors`):

| Processes | Gathered ranks | Received buffers in rank order | Gathered sums |
| --- | --- | --- | --- |
| 1 | `[0]` | `[[0, 1]]` | `[1]` |
| 2 | `[0, 1]` | `[[0, 1], [2, 3]]` | `[1, 5]` |
| 4 | `[0, 1, 2, 3]` | `[[0, 1], [2, 3], [4, 5], [6, 7]]` | `[1, 5, 9, 13]` |

## Graph format and sequential Dijkstra

`graph_utils.py` supplies shared validation, generation, loading, and saving.
Matrices are nonempty square, C-contiguous NumPy `float64` arrays. Entry `[u, v]`
is the weight of the directed edge from `u` to `v`; symmetric matrices describe
undirected graphs.

- `inf` / `np.inf` means **no edge**. Unreachable distances stay `inf`.
- The diagonal must be zero. An off-diagonal zero is a valid zero-cost edge.
- Negative weights anywhere in the graph, negative infinity, NaN, complex values,
  nonnumeric values, and malformed shapes are rejected.
- `.txt` files contain whitespace-separated rows with `inf` for missing edges.
  `.npy` files use NumPy's binary format, loaded with `allow_pickle=False`.
- Text saving uses 17 significant digits to preserve float64 weights.

`dijkstra_sequential.dijkstra(matrix)` returns all distances from node 0 without
modifying its input. It uses a linear scan to choose the next vertex and relaxes
that vertex's outgoing row: O(V²) time, O(V) working arrays, plus the validated
O(V²) matrix copy. It stops when no unsettled vertex is reachable. Ties are
resolved by vertex index. Distances use float64; a reachable shortest distance
outside its range raises `OverflowError` rather than being labeled unreachable.

Run the supplied example, which includes a zero-cost edge and an isolated vertex:

```powershell
.\.venv\Scripts\python.exe .\dijkstra_sequential.py --graph .\examples\zero_weight_disconnected.txt
```

Expected and observed distances: `[0, 0, 2, 3, inf]`. The program prints one line
per vertex, labeling `inf` as unreachable.

Generate a repeatable graph, save it, reload it, and convert it to text:

```powershell
New-Item -ItemType Directory -Force .\generated | Out-Null
.\.venv\Scripts\python.exe .\dijkstra_sequential.py --vertices 6 --density 0.6 --max-weight 9 --seed 42 --save-graph .\generated\graph.npy
.\.venv\Scripts\python.exe .\dijkstra_sequential.py --graph .\generated\graph.npy --save-graph .\generated\graph.txt
.\.venv\Scripts\python.exe .\dijkstra_sequential.py --graph .\generated\graph.txt
```

All three runs produced `[0, 2, 4, 9, 2, 2]`. Generated graphs are undirected by
default; add `--directed` for independent directed edges. Density is the
probability of each possible edge, and present weights are uniform integers from
0 through `--max-weight`, inclusive. Connectivity is not forced. Generation
options apply when `--graph` is omitted. Defaults are 5 vertices, density 0.5,
maximum weight 10, and seed 0. `generated/` is ignored by Git.

## Parallel Dijkstra

`dijkstra_parallel.dijkstra(matrix, comm=MPI.COMM_WORLD)` is a collective call:
rank 0 supplies the complete graph, and all other ranks supply `None`. It returns
the distance array on root 0 and `None` on workers. Root alone loads or generates
the graph when using the command-line interface. The shared graph helpers and
sequential reference are reused unchanged.

For `V` vertices and `P` processes, let `block = ceil(V / P)`. Rank `r` owns
vertices from `r * block` through `min((r + 1) * block, V) - 1`. Root packs the
graph's destination-column blocks into one C-contiguous `float64` NumPy buffer of
shape `(P, V, block)`. Uppercase `comm.Scatter()` sends an equal-sized `(V, block)`
buffer to every rank, with `local_weights[u, j] = graph[u, r * block + j]` for
real vertices. Thus each rank receives all incoming edges for its owned vertices,
including the original directed edge orientation. Extra columns contain `inf`
and are excluded from minimum selection and relaxation. This handles uneven
partitions and ranks with no vertices when `P > V`, while retaining `Scatter()`.

Each rank maintains distances and visited flags only for its own block. Every
iteration it finds its closest unvisited vertex and uses
`comm.allreduce((distance, global_vertex_id), op=MPI.MINLOC)` to select the global
choice. Equal distances choose the lowest vertex ID, matching the sequential
reference. Every rank relaxes its own destinations using the selected source row
of its local matrix; those neighbor weights are contiguous in memory. All ranks
stop together if the global minimum is infinite, leaving unreachable distances
as `inf`. Overflow checks are deferred so a later finite alternative can replace
an overflowing route, then combined collectively before raising `OverflowError`.

Lowercase `comm.gather()` collects the padded local distance arrays in rank order.
Root concatenates them and truncates to `V` entries, restoring vertex order and
removing padding. Root validates the graph and broadcasts any error before graph
distribution. Root also broadcasts command-line parsing, loading, generation, and
saving outcomes, including `--help`, so invalid input does not leave workers
waiting. Unexpected command-line failures call `MPI.COMM_WORLD.Abort()`.

Run the same five-vertex example with one, two, and four processes:

```powershell
.\.venv\Library\bin\mpiexec.exe -n 1 .\.venv\Scripts\python.exe .\dijkstra_parallel.py --graph .\examples\zero_weight_disconnected.txt --verify
.\.venv\Library\bin\mpiexec.exe -n 2 .\.venv\Scripts\python.exe .\dijkstra_parallel.py --graph .\examples\zero_weight_disconnected.txt --verify
.\.venv\Library\bin\mpiexec.exe -n 4 .\.venv\Scripts\python.exe .\dijkstra_parallel.py --graph .\examples\zero_weight_disconnected.txt --verify
```

The observed distances were `[0, 0, 2, 3, inf]` at every process count, with
`Verification: PASS`; two and four processes use uneven partitions. Generate a
directed graph with zero-weight edges and more processes
than vertices with:

```powershell
.\.venv\Library\bin\mpiexec.exe -n 4 .\.venv\Scripts\python.exe .\dijkstra_parallel.py --vertices 3 --density 1 --max-weight 0 --seed 42 --directed --verify
```

The observed distances were `[0, 0, 0]`, with `Verification: PASS`. The parallel
CLI accepts the same graph generation, loading, and `--save-graph` options as
the sequential CLI. `--verify`
compares every returned distance, including infinities, exactly with sequential
Dijkstra on the identical root graph, and prints `Verification: PASS` on success.
The option is off by default, and verification runs after the parallel solve.

The `_solve_local()` computation uses scalar minimum scans and relaxations like
the sequential baseline, with work distributed among ranks. Input validation,
packing, scattering, final gathering, and reference verification are outside that
kernel, so a later benchmarking phase can define comparable timing boundaries.
This phase makes no performance or speedup claims.

## Correctness checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

The Phase 2 regression run passed all **33 existing sequential/helper tests**.
Default discovery reported `Ran 39 tests in 0.087s: OK (skipped=6)`: the six
opt-in MPI CLI methods are skipped unless `RUN_MPI_TESTS=1` and were run
separately below. Fixed matrices have literal, independently calculated expected
distances, including:

| Case | Expected distances from 0 |
| --- | --- |
| Connected undirected graph | `[0, 3, 1, 4, 7]` |
| Disconnected component | `[0, 2, 5, inf, inf]` |
| Zero-weight directed edges | `[0, 0, 2, 3]` |
| Equal-cost alternative paths | `[0, 1, 1, 3]` |
| No edges | `[0, inf, inf]` |
| Single vertex | `[0]` |

Further checks cover directionality, fractional weights, input preservation,
negative weights in unreachable components, invalid matrices, reproducible graph
generation, zero/complete density, `.npy`/`.txt` round trips, singleton text
loading, and float64 path overflow. `pip check` reported no broken requirements.

Run the collective parallel API correctness suite at each process count:

```powershell
.\.venv\Library\bin\mpiexec.exe -n 1 .\.venv\Scripts\python.exe -m tests.mpi_dijkstra_cases
.\.venv\Library\bin\mpiexec.exe -n 2 .\.venv\Scripts\python.exe -m tests.mpi_dijkstra_cases
.\.venv\Library\bin\mpiexec.exe -n 4 .\.venv\Scripts\python.exe -m tests.mpi_dijkstra_cases
```

Observed Phase 2 results, with matching completion counts on every rank:

| Processes | Known graphs | Seeded random graphs | Recovery solves | Invalid inputs rejected | Overflow rejected | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 144 | 10 | 9 | 1 | PASS |
| 2 | 15 | 144 | 10 | 9 | 1 | PASS |
| 4 | 15 | 144 | 10 | 9 | 1 | PASS |

Every run compared all distances from **169 successful solves** against the
sequential reference: **507 successful solves** across the three process counts,
plus **30 expected input/overflow rejections**. Known graphs also use independent
literal expected distances. The cases cover directed and undirected graphs,
disconnected components, zero-weight edges and cycles, equal-cost paths, fractional
weights, noncontiguous input, input preservation, uneven partitions, and more ranks
than vertices. Random cases combine sizes `1, 2, 3, 5, 7, 8, 11, 17`, seeds
`0, 7, 42`, both orientations, and densities `0, 0.25, 1`; seed 0 uses exclusively
zero weights. Every rejected graph is followed by a successful solve on the same
communicator to verify recovery.

Run the opt-in MPI command-line integration tests from a normal PowerShell
session, rather than from inside `mpiexec`:

```powershell
$env:RUN_MPI_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest tests.test_parallel_cli -v
Remove-Item Env:RUN_MPI_TESTS
```

If reusing another MPI runtime, set `$env:MPIEXEC` to its matching launcher path
before running these tests. The six methods make **30 MPI launches** across one,
two, and four processes, checking example loading, generation with padding,
malformed/missing graph files, invalid generation and arguments, help, and
overflow. All **six methods and 30 launches passed** (`Ran 6 tests in 8.107s: OK`).
The 21 invalid-input/overflow launches returned exit code 2 with one root error
and no traceback; the six successful solves and three help launches returned
exit code 0. Each launch has a timeout that terminates its worker processes if
it hangs; no launch timed out.

During automated setup, the Windows sandbox blocked environment bootstrap,
package downloads, MPI worker creation, and default temporary-file access. The
affected commands succeeded when rerun with the tool's required access. The
initial temporary-file test errors were permission failures; the full suite
passed after that access was available. These commands run directly in a normal
VS Code PowerShell terminal.

## Files

| File | Purpose |
| --- | --- |
| `mpi_smoke.py` | Validated root-0 Scatter/gather smoke test |
| `dijkstra_sequential.py` | Sequential algorithm and command-line interface |
| `dijkstra_parallel.py` | Distributed column-block algorithm and root-0 MPI CLI |
| `graph_utils.py` | Shared adjacency-matrix helpers |
| `examples/zero_weight_disconnected.txt` | Small loadable example |
| `tests/test_dijkstra_sequential.py` | Known-distance and algorithm checks |
| `tests/test_graph_utils.py` | Validation, generation, and file checks |
| `tests/mpi_dijkstra_cases.py` | Collective known/random comparisons, rejection, and recovery checks |
| `tests/test_parallel_cli.py` | Opt-in MPI subprocess success/error checks with timeouts |
| `requirements.txt` | Tested NumPy and mpi4py versions |
| `.gitignore` | Excludes `.venv/`, caches, generated data, and crash dumps |
