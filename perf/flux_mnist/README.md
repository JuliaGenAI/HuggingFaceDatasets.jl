# Flux MNIST — data-loading benchmark (CPU)

Trains a small MLP on MNIST pulled from the HuggingFace `datasets` library, comparing
**data-loading strategies** (on-the-fly vs. materialized, thread- vs. process-parallel) in
Flux+HuggingFaceDatasets.jl against an equivalent PyTorch version. It's the tiny-model counterpart of
the [`flux_cifar10`](../flux_cifar10) example: same structure, but the MLP is so small that this
benchmark is almost purely about **data loading**, not compute.

## Files

| file | stack | notes |
| --- | --- | --- |
| [`flux_mnist.jl`](flux_mnist.jl) | Flux + HuggingFaceDatasets.jl | `julia --project=. -t4 flux_mnist.jl` |
| [`pytorch_mnist.py`](pytorch_mnist.py) | PyTorch + `datasets` | `uv run pytorch_mnist.py` |

## Setup

Same model and hyperparameters everywhere: MLP `784 → 100 → 100 → 10`, `AdamW(1e-3)`, cross-entropy,
batch size 128, **10 epochs**, **CPU only** (the model is tiny — keeping it off the GPU isolates the
data-loading cost, which is the point of this example). MNIST (`ylecun/mnist`) is pulled via HF
`datasets`; the Julia side uses **MLUtils 0.4.12**, whose `num_workers` path returns collated batches
through **shared memory**. Both frameworks reach ~**98%** test accuracy after 10 epochs.

Timings are single-run wall-clock on a **Ryzen Threadripper PRO 9955WX** (Julia `-t4`) — indicative,
not rigorous (expect run-to-run variation). Each script first runs a short verbose **DEMO** (accuracy
per epoch), then times each config twice: once for **full** training and once for **loader iteration
alone** (no model, same 10 epochs). **Every timed config discards one warm-up epoch before timing**, so
Julia's first-call JIT, worker-process startup, and the shm-session build — and PyTorch's
persistent-worker spawn — stay out of the numbers. These are steady-state per-epoch costs.

> **Threads + a PythonCall-backed dataset.** HuggingFaceDatasets shares numpy arrays into Julia
> *zero-copy*, keeping the batches backed by Python-owned buffers. Those buffers are released through
> PythonCall's GIL-deferred decref: a finalizer that can't take the GIL just enqueues the pointer, so
> one firing on a `parallel=true` worker thread never deadlocks. The materialized paths here decode the
> split into plain in-memory Julia arrays with `[:]` (no Python buffer left to touch), which is what
> lets `parallel=true` (threads) run cleanly on them — see the comments in
> [`flux_mnist.jl`](flux_mnist.jl). For on-the-fly PythonCall data, prefer `num_workers` (separate
> processes, separate GILs) over `parallel` (threads) to get real parallelism past the GIL.

## Results

### Julia — Flux + HuggingFaceDatasets.jl (`-t4`, MLUtils 0.4.12)

| config | data loading | full | load only |
| --- | --- | ---: | ---: |
| Serial | on-the-fly, `num_workers=0` | 23.0 s | 17.8 s |
| Serial Materialized | in-memory `[:]`, `num_workers=0` | 4.7 s | 0.4 s |
| Parallel Materialized | in-memory, `parallel=true` (threads) | 5.0 s | 0.3 s |
| Distributed | on-the-fly, `num_workers=2` (processes, shared-mem) | 13.3 s | 10.0 s |
| Distributed | on-the-fly, `num_workers=4` (processes, shared-mem) | 7.9 s | 5.4 s |
| Distributed | on-the-fly, `num_workers=8` (processes, shared-mem) | 5.5 s | 3.1 s |

### PyTorch (`pytorch_mnist.py`)

| config | data loading | full | load only |
| --- | --- | ---: | ---: |
| Serial | on-the-fly, `num_workers=0` | 17.9 s | 13.4 s |
| Serial Materialized | in-memory tensors, `num_workers=0` | 4.4 s | 1.3 s |
| Parallel Materialized | in-memory, `num_workers=4` (processes) | 5.2 s | 1.2 s |
| Distributed | on-the-fly, `num_workers=2` (processes) | 9.8 s | 8.3 s |
| Distributed | on-the-fly, `num_workers=4` (processes) | 6.2 s | 4.9 s |
| Distributed | on-the-fly, `num_workers=8` (processes) | 6.4 s | 2.8 s |

## Takeaways

- **For data this small, materialize and you're done.** Decoding all 60k images once drops per-epoch
  loading to essentially nothing — `load only` **17.8 → 0.4 s** in Julia, **13.4 → 1.3 s** in PyTorch —
  because MNIST is ~180 MB and lives comfortably in RAM. Materialized is the fastest full-training
  config in both frameworks (Julia 4.7 s, PyTorch 4.4 s).
- **`num_workers` now scales — it's just unnecessary here.** With the shared-memory transport, Julia's
  Distributed `load only` drops **17.8 → 10.0 → 5.4 → 3.1 s** at `num_workers` 0/2/4/8, tracking PyTorch
  (13.4 → 8.3 → 4.9 → 2.8 s) almost step for step. This is a real change from the past, when process
  workers were *slower* than serial for a model this small. But worker IPC/startup still can't beat a
  materialized in-memory scan (0.3–1.3 s), so for a dataset that fits in memory, **materialize beats
  `num_workers`** — reach for workers only when the data *doesn't* fit and the per-epoch decode must be
  parallelized.
- **The MLP is too small for compute to matter — this is a loading benchmark.** `full − load` is ~3–5 s
  everywhere (the entire train step for a 784→100→100→10 MLP is nearly free), so the two frameworks land
  close together across the board. Julia's materialized iteration is faster (`load only` 0.4 vs 1.3 s);
  PyTorch's on-the-fly serial is faster (13.4 vs 17.8 s). Effectively a wash — as expected when there's
  almost nothing to compute. For the case where compute *does* matter, see the
  [CIFAR-10 example](../flux_cifar10).
- **`parallel=true` (threads) needs a pure-Julia dataset.** It shines on the materialized path
  (`load only` 0.3 s, edging serial-materialized — there's nothing left to parallelize once the data
  is in memory) but only because the materialize step copies the data out of Python first. Threads over
  a live PythonCall-backed dataset would still be GIL-serialized on every decode (though, thanks to the
  deferred decref, they no longer deadlock — see the note above); that's what `num_workers` (processes)
  is for.
