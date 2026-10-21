# HPC Assignment 3: first implementation phase

Native Windows / VS Code PowerShell setup, an MPI smoke test, and sequential
adjacency-matrix Dijkstra from **source vertex 0**. The parallel Dijkstra,
performance experiments, plots, and PDF report belong to the next phase.

The assignment requires an adjacency matrix, uppercase `comm.Scatter()` for graph
distribution, and lowercase `comm.gather()` for results in the eventual parallel
implementation. The smoke test follows Lecture 7's `MPI_COMM_WORLD`, rank/size,
root-0, scatter/local-sum/gather examples (pages 4, 28–31, and 41). It checks the
collectives using small integer buffers; it does not yet distribute a graph.

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

## Correctness checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

The final suite passed **33 tests** in 0.131 seconds. Fixed matrices have literal,
independently calculated expected distances, including:

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
| `graph_utils.py` | Shared adjacency-matrix helpers |
| `examples/zero_weight_disconnected.txt` | Small loadable example |
| `tests/test_dijkstra_sequential.py` | Known-distance and algorithm checks |
| `tests/test_graph_utils.py` | Validation, generation, and file checks |
| `requirements.txt` | Tested NumPy and mpi4py versions |
| `.gitignore` | Excludes `.venv/`, caches, generated data, and crash dumps |
