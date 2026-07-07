# Flux CIFAR-10 — CNN + data-loading benchmark (GPU)

Trains a small VGG-style CNN on CIFAR-10 pulled from the HuggingFace `datasets` library, with the
standard crop+flip augmentation, **on the GPU**. It compares **data-loading strategies** (on-the-fly
vs. materialized, thread- vs. process-parallel) in Flux+HuggingFaceDatasets.jl against an equivalent
PyTorch version — the CIFAR-10 counterpart of the [`flux_mnist`](../flux_mnist) example.

## Files

| file | stack | notes |
| --- | --- | --- |
| [`flux_cifar10.jl`](flux_cifar10.jl) | Flux + HuggingFaceDatasets.jl | `julia --project=. -t4 flux_cifar10.jl` |
| [`pytorch_cifar10.py`](pytorch_cifar10.py) | PyTorch + `datasets` | `uv run pytorch_cifar10.py` |

## Setup

Same model and hyperparameters everywhere: a VGG-style CNN — three `conv-conv-pool` blocks
(64→128→256 channels, `3×3` convs + BatchNorm), then `Dense(4096→256) → Dropout(0.5) → Dense(256→10)`
(~2.2M params) — with `AdamW(1e-3)`, cross-entropy, batch size 128, **10 epochs**, on the **GPU**.
CIFAR-10 (`uoft-cs/cifar10`) is pulled via HF `datasets`. Augmentation is the classic pipeline:
per-channel **normalize**, random **crop** to `32×32` with 4-px zero padding, and a random
**horizontal flip** (test data is only normalized). The Julia side uses **MLUtils 0.4.12**, whose
`num_workers` path returns collated batches through **shared memory** — see the Distributed rows.

Timings are single-run wall-clock on an **RTX 5090** (CPU: Ryzen Threadripper PRO 9955WX) —
indicative, not rigorous (expect run-to-run variation). Each script first runs a short verbose
**DEMO** (accuracy per epoch), then times each config twice: once for **full** training and once for
**loader iteration alone** (no model, same 10 epochs). **Every timed config discards one warm-up
epoch before timing**, so Julia's first-call JIT, worker-process startup, and the shm-session build —
and PyTorch's persistent-worker spawn — stay out of the numbers. These are steady-state per-epoch
costs. Both frameworks reach ~**80–83%** test accuracy after 10 epochs — this is a data-loading
timing benchmark, not a numerical match (init/weight-decay details differ), so accuracies only track
loosely.

In the tables below, **full** = train + load, and **load only** = iterate the `DataLoader` for the
same 10 epochs consuming each batch but running no model. For the serial paths (no prefetch) `full ≈
load + compute`, so `full − load` is roughly the GPU step; for the worker/thread paths loading
overlaps compute, so `full` collapses toward whichever of the two is larger.

## Results

### Julia — Flux + HuggingFaceDatasets.jl (`-t4`, MLUtils 0.4.12)

| config | data loading | full | load only |
| --- | --- | ---: | ---: |
| Serial | on-the-fly, `num_workers=0` | 46.9 s | 33.6 s |
| Serial Materialized | in-memory `[:]`, `num_workers=0` | 16.4 s | 5.5 s |
| Parallel Materialized | in-memory, `parallel=true` (threads) | 14.5 s | 3.3 s |
| Distributed | on-the-fly, `num_workers=2` (processes, shared-mem) | 23.7 s | 20.6 s |
| Distributed | on-the-fly, `num_workers=4` (processes, shared-mem) | 17.4 s | 11.8 s |
| Distributed | on-the-fly, `num_workers=8` (processes, shared-mem) | 16.4 s | 7.4 s |

### PyTorch (`pytorch_cifar10.py`)

| config | data loading | full | load only |
| --- | --- | ---: | ---: |
| Serial | on-the-fly, `num_workers=0` | 53.0 s | 40.1 s |
| Serial Materialized | in-memory tensors, `num_workers=0` | 16.5 s | 10.1 s |
| Parallel Materialized | in-memory, `num_workers=4` (processes) | 11.5 s | 3.9 s |
| Distributed | on-the-fly, `num_workers=2` (processes) | 22.0 s | 22.1 s |
| Distributed | on-the-fly, `num_workers=4` (processes) | 13.0 s | 12.3 s |
| Distributed | on-the-fly, `num_workers=8` (processes) | 11.9 s | 7.2 s |

<sub>The on-the-fly rows use the standard torchvision order (augment the PIL image, then
`ToTensor`+`Normalize`); the materialized rows augment already-normalized CHW tensors, matching
Julia's decode/augment split. The pixels differ slightly, but each is the idiomatic path in its
framework, so the per-item work is representative.</sub>

## Takeaways

- **`num_workers` now scales in Julia — the shared-memory transport (MLUtils 0.4.12).** Julia's
  Distributed `load only` drops **33.6 → 20.6 → 11.8 → 7.4 s** at `num_workers` 0/2/4/8 (1.6× / 2.8× /
  4.5×), tracking PyTorch's own worker curve almost exactly (**40.1 → 22.1 → 12.3 → 7.2 s**). Before
  0.4.12 each collated batch came back through a serialize→socket→deserialize round-trip and the curve
  was flat — more workers didn't help. Now only a shared-memory *handle* crosses the socket (the ~1.5 MB
  of pixels stays put), so the process-parallel path parallelizes for real. **Julia's distributed loader
  is now competitive with PyTorch's.**
- **Julia loads at least as fast as PyTorch, and faster off the distributed path.** Across `load
  only`: serial 33.6 vs 40.1 s, materialized 5.5 vs 10.1 s, threaded 3.3 vs 3.9 s; distributed is a
  tie (11.8 vs 12.3; 7.4 vs 7.2). Less per-item Python overhead reaching the Arrow rows.
- **PyTorch still computes this CNN faster.** Once loading is cheap or fully hidden, the GPU step
  dominates and PyTorch's mature cuDNN path is quicker: the full-training floor is ~**11.5 s**
  (Parallel Materialized) vs Julia's ~**14.5 s**. So PyTorch edges the full-training totals on the
  parallel paths (Distributed 13.0 vs 17.4 s at 4 workers) even though its loading is marginally slower —
  the two frameworks pull in opposite directions, but the gap is small and it's now the *compute*, not
  the loading, that decides it.
- **When the data fits in RAM, materialize + threads is Julia's best path.** Decoding all 50k PNGs
  once drops per-epoch loading to near-nothing (`load only` 5.5 s materialized; **3.3 s** with
  `parallel=true`), and `parallel=true` uses **threads** — shared memory, no IPC, no serialization at
  all — so it is both the fastest loading path here and the fastest Julia full-training config
  (14.5 s). PyTorch can't thread the loader (GIL), so its in-memory-parallel path uses worker
  *processes* (3.9 / 11.5 s). Rule of thumb: **materialize + `parallel=true`** when the set fits in
  memory; reach for **`num_workers`** when it doesn't and the per-epoch decode must be parallelized
  past the GIL.
