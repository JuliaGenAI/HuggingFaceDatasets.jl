# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "datasets", "numpy", "pillow"]
# ///
#
# PyTorch counterpart of flux_mnist.jl (Flux + HuggingFaceDatasets), for a data-loading timing
# comparison on the same task: an MLP on MNIST pulled from the same HuggingFace `datasets` Arrow
# dataset, on the CPU. Run with:  uv run perf/flux_mnist/pytorch_mnist.py
#
# The configs mirror flux_mnist.jl. The on-the-fly path uses the idiomatic HF pattern
# (`Dataset.with_transform` + a `collate_fn`); the materialized path decodes the split into in-memory
# tensors once. One caveat: PyTorch's only DataLoader parallelism is multiprocess (`num_workers`), so
# the "Parallel Materialized" row uses worker *processes* over in-memory tensors — Julia's
# `parallel=true` there uses *threads* (cheap, shared memory), which PyTorch cannot do because of the
# GIL. This is a timing benchmark, not a numerical match: the exact AdamW weight decay etc. differ,
# so accuracies only track loosely.

import os
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, disable_progress_bars
from datasets.utils.logging import set_verbosity_error
from torch.utils.data import DataLoader, TensorDataset

disable_progress_bars()
set_verbosity_error()

BATCHSIZE = 128
NHIDDEN = 100
EPOCHS = int(os.environ.get("EPOCHS", "4"))
DEVICE = torch.device("cpu")


def make_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, NHIDDEN), nn.ReLU(),
        nn.Linear(NHIDDEN, NHIDDEN), nn.ReLU(),
        nn.Linear(NHIDDEN, 10),
    ).to(DEVICE)


def lazy_transform(batch):
    """Decode + rescale on access (registered with `with_transform`); picklable to workers.
    Matches Julia's `image ./ 255f0` (plain [0, 1] scaling, no normalization)."""
    batch["pixel_values"] = [np.asarray(img, dtype=np.float32) / 255.0 for img in batch["image"]]
    return batch


def collate(examples):
    x = torch.from_numpy(np.stack([e["pixel_values"] for e in examples]))
    y = torch.tensor([e["label"] for e in examples])
    return x, y


def materialize(hf_ds):
    """Decode the whole split into in-memory tensors up front (the `[:]` path in flux_mnist.jl)."""
    ds = hf_ds.with_format("numpy")
    col = ds["image"]
    images = col if isinstance(col, np.ndarray) else np.stack(col)
    images = torch.from_numpy(images.astype(np.float32) / 255.0)          # (N, 28, 28)
    labels = torch.from_numpy(np.asarray(ds["label"]).astype(np.int64))   # (N,)
    return TensorDataset(images, labels)


def make_loaders(num_workers, materialize_data):
    if materialize_data:
        train_ds = materialize(load_dataset("ylecun/mnist", split="train"))
        test_ds = materialize(load_dataset("ylecun/mnist", split="test"))
        collate_fn = None
    else:
        ds = load_dataset("ylecun/mnist")
        train_ds = ds["train"].with_transform(lazy_transform)
        test_ds = ds["test"].with_transform(lazy_transform)
        collate_fn = collate

    # persistent_workers keeps the pool alive across epochs (closer to Julia's leased pool) so the
    # discarded warm-up epoch below pays the worker-spawn cost once, out of the timed region.
    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, **kw)
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, **kw)
    return train_loader, test_loader


@torch.no_grad()
def loss_and_accuracy(loader, model):
    model.eval()
    lossfn = nn.CrossEntropyLoss(reduction="sum")
    ls, correct, num = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        ls += lossfn(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        num += y.shape[0]
    return ls / num, correct / num


def train(num_workers=0, materialize_data=False, verbose=False, loader_only=False):
    """Return wall-clock seconds of EPOCHS timed epochs; one warm-up epoch runs first and is discarded
    (spawns the persistent workers), so worker-spawn cost stays out of the timing. `verbose=True` is
    the DEMO path: report accuracy per epoch instead, no warm-up, no timing."""
    train_loader, test_loader = make_loaders(num_workers, materialize_data)

    if loader_only:
        for _ in train_loader:                 # warm-up (discarded)
            pass
        seen = 0
        t0 = time.perf_counter()
        for _ in range(EPOCHS):
            for x, y in train_loader:
                seen += y.shape[0]
        return time.perf_counter() - t0

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossfn = nn.CrossEntropyLoss()

    def run_epoch():
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossfn(model(x), y)
            loss.backward()
            opt.step()

    if verbose:
        def report(epoch):
            tr_loss, tr_acc = loss_and_accuracy(train_loader, model)
            te_loss, te_acc = loss_and_accuracy(test_loader, model)
            print(f"(epoch = {epoch}, train_loss = {tr_loss:.3f}, train_acc = {tr_acc:.3f}, "
                  f"test_loss = {te_loss:.3f}, test_acc = {te_acc:.3f})")
        report(0)
        for epoch in range(1, EPOCHS + 1):
            run_epoch()
            report(epoch)
        return

    run_epoch()                                # warm-up (discarded)
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        run_epoch()
    return time.perf_counter() - t0


def timed(name, **kw):
    t = train(**kw)
    print(f"### {name}\n  {t:.1f} seconds  ({EPOCHS} epochs, warm-up discarded)")


if __name__ == "__main__":
    print("### DEMO — MLP learning MNIST on CPU (accuracy per epoch)")
    train(num_workers=0, materialize_data=True, verbose=True)

    # Each config runs one warm-up epoch (discarded) before its timed epochs, so worker-spawn stays
    # out of the numbers — matching the Julia script's warm-up discipline.
    print("\n#### FULL TRAINING — model + data loading ###########")
    timed("Serial", num_workers=0, materialize_data=False)
    timed("Serial Materialized", num_workers=0, materialize_data=True)
    timed("Parallel Materialized", num_workers=4, materialize_data=True)
    timed("Distributed (2 workers)", num_workers=2, materialize_data=False)
    timed("Distributed (4 workers)", num_workers=4, materialize_data=False)
    timed("Distributed (8 workers)", num_workers=8, materialize_data=False)
    print("#### END FULL TRAINING ###########")

    # Same configs, but iterating the loader with no model — the pure data-loading cost.
    print(f"\n#### DATA-LOADING ONLY — no model, same {EPOCHS} epochs ###########")
    timed("Serial", num_workers=0, materialize_data=False, loader_only=True)
    timed("Serial Materialized", num_workers=0, materialize_data=True, loader_only=True)
    timed("Parallel Materialized", num_workers=4, materialize_data=True, loader_only=True)
    timed("Distributed (2 workers)", num_workers=2, materialize_data=False, loader_only=True)
    timed("Distributed (4 workers)", num_workers=4, materialize_data=False, loader_only=True)
    timed("Distributed (8 workers)", num_workers=8, materialize_data=False, loader_only=True)
    print("#### END DATA-LOADING ONLY ###########")
