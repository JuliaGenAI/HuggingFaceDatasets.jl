"""Python data-access benchmarks for the Hugging Face ``datasets`` library.

Mirror of ``perf.jl``: the same deep-learning input-pipeline operations on the MNIST
test split (10k x 28x28 uint8 images), run straight through ``datasets`` so the Julia
and Python tables are directly comparable.

    single   per-observation access:  ds[i]        for every i
    batch    batched access:          ds[i:i+128]  over the whole split
    full     materialize everything:  ds[:]
    epoch    a realistic pass:        batch + cast to float32/255 + one-hot labels

across the two formats a PyTorch/NumPy pipeline would use:

    plain    no format set          (raw Python objects / PIL images)
    numpy    ds.with_format("numpy")  (NumPy arrays)

The ``epoch`` task needs numeric arrays, so it is only run on the ``numpy`` format
(matching the Julia ``julia``-format row). Run with this folder's Conda env, e.g.

    julia --project=perf -e 'using CondaPkg; run(`$(CondaPkg.which("python")) perf/perf.py`)'
"""

import time

import numpy as np
from datasets import load_dataset

BS = 128


def timed(fn, repeats=5):
    """Minimum wall-clock over `repeats` runs, in milliseconds (BenchmarkTools-style)."""
    fn()  # warm up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def single(ds):
    for i in range(len(ds)):
        ds[i]


def batch(ds):
    n = len(ds)
    for i in range(0, n, BS):
        ds[i : min(i + BS, n)]


def full(ds):
    ds[:]


def epoch(ds):
    n = len(ds)
    s = np.float32(0)
    for i in range(0, n, BS):
        b = ds[i : min(i + BS, n)]
        x = b["image"].astype(np.float32) / 255.0
        labels = b["label"]
        y = np.zeros((10, len(labels)), dtype=np.float32)
        y[labels, np.arange(len(labels))] = 1.0
        s += x.sum() + y.sum()
    return s


def main():
    base = load_dataset("ylecun/mnist", split="test")
    plain = base.with_format(None)
    numpy = base.with_format("numpy")

    variants = [
        ("plain", plain, False),
        ("numpy", numpy, True),
    ]

    rows = []
    for name, ds, do_epoch in variants:
        print(f"benchmarking {name} ...")
        t_single = timed(lambda: single(ds))
        t_batch = timed(lambda: batch(ds))
        t_full = timed(lambda: full(ds))
        t_epoch = timed(lambda: epoch(ds)) if do_epoch else None
        rows.append((name, t_single, t_batch, t_full, t_epoch))

    def cell(x):
        return f"{'—':>10}" if x is None else f"{x:10.1f}"

    print("\n## Python (datasets) — MNIST test, times in ms\n")
    print(f"| {'variant':<11} | {'single':>10} | {'batch':>10} | {'full':>10} | {'epoch':>10} |")
    print(f"|{'-'*13}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")
    for name, s, b, f, e in rows:
        print(f"| {name:<11} | {cell(s)} | {cell(b)} | {cell(f)} | {cell(e)} |")


if __name__ == "__main__":
    main()
