# Performance

Benchmarks comparing the Julia wrapper (HuggingFaceDatasets.jl) against the underlying
Python [`datasets`](https://huggingface.co/docs/datasets) library on the data-loading and
preprocessing operations that dominate a deep-learning input pipeline.

Both scripts run the **same tasks** on the **MNIST test split** (10 000 × 28×28 `UInt8`
images), so the two result tables are directly comparable.

| task     | what it measures                                                    |
| -------- | ------------------------------------------------------------------- |
| `single` | per-observation access — `getobs(ds, i)` / `ds[i]` for every `i`    |
| `batch`  | batched access — `ds[i:i+127]` over the whole split (batch size 128)|
| `full`   | materialize everything at once — `ds[:]`                            |
| `epoch`  | a realistic pass — batch + cast to `Float32`/255 + one-hot labels   |

## Files

- [`perf.jl`](perf.jl) — Julia benchmarks (uses `BenchmarkTools`, minimum time reported).
- [`perf.py`](perf.py) — Python benchmarks through `datasets` (minimum of 5 runs).
- [`Project.toml`](Project.toml) — the Julia environment for `perf.jl`.

## Running

```bash
# Julia side (instantiates on first run)
julia --project=perf perf/perf.jl

# Python side, using this project's Conda environment
julia --project=perf -e 'using CondaPkg; run(`$(CondaPkg.which("python")) perf/perf.py`)'
```

## Variants

**Julia** (`perf.jl`):

| variant      | how                                    | observations are …                    |
| ------------ | -------------------------------------- | ------------------------------------- |
| `mldatasets` | `MLDatasets.MNIST`                     | native Julia arrays, fully in memory  |
| `plain`      | `set_format!(ds, nothing)`             | raw Python objects / PIL images       |
| `numpy`      | `with_format(ds, "numpy")`             | raw NumPy arrays (no `py2jl`)          |
| `julia`      | `with_format(ds, "julia")` *(default)* | native Julia arrays (numpy + `py2jl`) |

**Python** (`perf.py`):

| variant | how                        | observations are …             |
| ------- | -------------------------- | ------------------------------ |
| `plain` | no format set              | raw Python objects / PIL images|
| `numpy` | `ds.with_format("numpy")`  | NumPy arrays                   |

The `epoch` task needs numeric arrays, so it is only run on the array-producing variants
(`mldatasets` and `julia` in Julia; `numpy` in Python). For `plain`/`numpy` in Julia the
batch is a raw Python object, and converting it to Julia arrays is exactly what the `julia`
format already does.

## Results

Measured on an M1 Pro (macOS, arm64). Times in **milliseconds** (lower is better).
Software: Julia 1.12, Python 3.12, `datasets` 4.8.5, NumPy 2.5, PyArrow 24.

### Julia — HuggingFaceDatasets.jl

| variant     |     single |      batch |       full |      epoch |
|-------------|-----------:|-----------:|-----------:|-----------:|
| mldatasets  |        7.9 |        2.8 |        2.7 |        5.0 |
| plain       |      720.1 |      320.0 |      309.7 |          — |
| numpy       |     1546.0 |      378.1 |      394.8 |          — |
| julia       |     1936.7 |      377.6 |      381.8 |      370.9 |

### Python — datasets

| variant     |     single |      batch |       full |      epoch |
|-------------|-----------:|-----------:|-----------:|-----------:|
| plain       |      415.8 |      246.8 |      277.2 |          — |
| numpy       |      965.4 |      285.2 |      311.8 |      309.7 |

### Parallel loading — process pool (HuggingFaceDatasets.jl)

Reading the whole split in batches (`getobs(ds, i:j)`, batch size 128), serial vs across N
worker **processes** (`Distributed`). Thread-based `parallel=true` cannot help — the CPython
GIL serializes the read — but separate processes each hold their own interpreter/GIL, and a
`Dataset` serializes as a by-reference pickle of its on-disk Arrow files (workers re-mmap, no
copy), so the reads run concurrently. Single run; the `serial` row is the in-run baseline.

| workers   |  time (ms) |  speedup |
|-----------|-----------:|---------:|
| serial    |      265.5 |    1.00× |
| 2 procs   |      151.8 |    1.75× |
| 4 procs   |       83.2 |    3.19× |

## Takeaways

- **Batch, don't loop over samples.** Per-observation access is the slowest path in every
  variant: each `ds[i]` decodes one Arrow row (and, in Julia, crosses the PythonCall FFI
  boundary), and doing that 10 000 times dominates. Batched access is ~5× faster in Julia
  and ~3× faster in Python. Prefer `getobs(ds, i:j)` / `ds[:]`.

- **The wrapper's overhead is small at batch granularity.** For the realistic `epoch`
  pipeline the `julia` format costs ~370 ms vs ~310 ms for Python's `numpy` format — about
  **1.2×**. The convenience of native Julia arrays (copyless, zero-copy `py2jl`) is nearly
  free once you work in batches. Per-sample access is where the FFI cost shows up
  (`single`: ~2× Python), so avoid it in hot loops.

- **Arrow-backed access is the bottleneck, not Julia.** `MLDatasets` — native, fully
  in-memory — is ~100× faster than any `datasets`-backed path. When a dataset fits in
  memory, materialize it once with `getobs(ds, :)` and feed an in-memory `DataLoader`
  (see [`../flux_mnist/flux_mnist.jl`](../flux_mnist/flux_mnist.jl)); the Arrow decode then
  becomes a one-time cost rather than a per-batch one.

- **When it doesn't fit in memory, parallelize the read with processes, not threads.** The
  Arrow-decode/`py2jl` read is GIL-bound, so a thread pool (`DataLoader(parallel=true)`)
  reads no faster than one thread. Worker *processes* each get their own interpreter/GIL and
  scale — ~3× on 4 processes for the MNIST batch read above. A `Dataset` pickles by reference
  to its on-disk Arrow files, so workers share the data (re-mmap, no copy); this is what a
  process-based `DataLoader(ds; num_workers=N)` would build on.
