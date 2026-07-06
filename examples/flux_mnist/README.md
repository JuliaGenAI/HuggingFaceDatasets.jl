# Flux MNIST — data-loading benchmark

Trains a small MLP on MNIST pulled from the HuggingFace `datasets` library, comparing
**data-loading strategies** (on-the-fly vs. materialized, thread- vs. process-parallel) in
Flux+HuggingFaceDatasets.jl against equivalent and idiomatic PyTorch versions.

## Files

| file | stack | notes |
| --- | --- | --- |
| [`flux_mnist.jl`](flux_mnist.jl) | Flux + HuggingFaceDatasets.jl | `julia --project=. -t4 flux_mnist.jl` |
| [`pytorch_mnist.py`](pytorch_mnist.py) | PyTorch + `datasets` | 1:1 port of `flux_mnist.jl`; `uv run pytorch_mnist.py` |
| [`pytorch_mnist_idiomatic.py`](pytorch_mnist_idiomatic.py) | PyTorch + `datasets` | idiomatic HF patterns (`with_transform`, `.map`); `uv run pytorch_mnist_idiomatic.py` |

## Setup

Same model and hyperparameters everywhere: MLP `784 → 100 → 100 → 10`, `AdamW(1e-3)`,
cross-entropy, batch size 128, **4 epochs**, **CPU only**, MNIST (`ylecun/mnist`) via HF
`datasets`. Timings are single-run wall-clock on an **Apple M1 Pro** — indicative, not
rigorous (expect ±10–20% run to run). A warm-up epoch precedes the Julia timings to exclude
Julia's one-time JIT compilation; PyTorch runs eagerly (no comparable compile step) so needs
none. Both then measure steady-state per-epoch compute — imports and process startup sit
outside the timed region either way — so the tables are directly comparable.

## Results

### Julia — Flux + HuggingFaceDatasets.jl (`-t4`)

| config | data loading | time |
| --- | --- | ---: |
| Serial | on-the-fly, `num_workers=0` | 13.5 s |
| Serial Materialized | in-memory `[:]`, `num_workers=0` | 6.4 s |
| Parallel Materialized | in-memory, `parallel=true` (threads) | 9.1 s |
| Distributed | on-the-fly, `num_workers=4` (processes) | 33.0 s |

<sub>A warm-up epoch runs first, so these exclude Julia's JIT compilation. The serial path is
then fully warmed; the `parallel`/`num_workers` paths still compile their first call, and
Distributed also pays worker-process startup — real costs for a short job, left in the numbers.</sub>

### PyTorch — 1:1 port (`pytorch_mnist.py`)

| config | data loading | time |
| --- | --- | ---: |
| Serial | on-the-fly, `num_workers=0` | 62.8 s |
| Serial Materialized | in-memory tensors, `num_workers=0` | 16.8 s |
| Parallel Materialized | in-memory, `num_workers=4` (processes) | 22.2 s |
| Distributed | on-the-fly, `num_workers=4` (processes) | 66.6 s |

### PyTorch — idiomatic HF (`pytorch_mnist_idiomatic.py`)

| config | data loading | time |
| --- | --- | ---: |
| Lazy `with_transform` | on-the-fly, `num_workers=0` | 34.1 s |
| Lazy `with_transform` | on-the-fly, `num_workers=4` | 55.1 s |
| Cached `.map` (torch format) | preprocessed to Arrow, `num_workers=0` | 291.4 s† |
| Cached `.map` (torch format) | preprocessed to Arrow, `num_workers=4` | 128.3 s |

<sub>†includes the one-time `.map` decode+cache; the `num_workers=4` row reuses that cache.</sub>

## Takeaways

- **Materializing into memory is the biggest win** for a dataset this small — ~2× in Julia
  (13.5 → 6.4 s) and ~4× in PyTorch (62.8 → 16.8 s), because it drops the per-batch CPython
  decode entirely.
- **Parallel loading does *not* pay off here.** The MLP is tiny, so multiprocess
  pickling/IPC overhead outweighs the parallel-decode benefit — `num_workers=4` was *slower*
  than `num_workers=0` in every PyTorch case, and Julia's `Distributed` (33.0 s) lost to its
  serial on-the-fly (13.5 s). `num_workers`/process parallelism is for when `getobs` is the
  bottleneck (large images, heavy decode), not toy workloads.
  - Julia's **thread**-based `parallel=true` over *materialized* data (9.1 s) is cheap
    (shared memory, needs `-t>1`) but pointless here — there's nothing to parallelize once the
    data is plain in-memory arrays. PyTorch has no thread analog (the GIL); its only knob is
    `num_workers` (processes).
- **Idiomatic PyTorch is competitive but has a trap.** Lazy `with_transform` (34.1 s) even
  beats the 1:1 Serial port (62.8 s), but naively `.map`-ing decoded float tensors into Arrow
  and reading them back via the torch formatter is *much* slower (128–291 s) — for images,
  prefer lazy transforms or materialize to plain tensors, don't cache decoded floats.
- **Julia came out ahead of PyTorch** on the like-for-like paths (on-the-fly 13.5 vs 62.8 s;
  materialized 6.4 vs 16.8 s) — a fair comparison, since both are steady-state per-epoch
  compute (Julia after a warm-up, PyTorch eager with no compile step to warm away).
