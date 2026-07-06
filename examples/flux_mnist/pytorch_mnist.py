# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "datasets", "numpy", "pillow"]
# ///
#
# PyTorch counterpart of flux_mnist.jl (Flux + HuggingFaceDatasets), for a data-loading timing
# comparison on the same task: an MLP on MNIST pulled from the same HuggingFace `datasets`
# Arrow dataset. Run with:  uv run examples/flux_mnist/pytorch_mnist.py
#
# The four configs mirror flux_mnist.jl. One caveat: PyTorch's only DataLoader parallelism is
# multiprocess (`num_workers`), so the "Parallel Materialized" row uses worker *processes*
# over in-memory tensors — Julia's `parallel=true` there uses *threads* (cheap, shared
# memory), which PyTorch cannot do because of the GIL. This is a timing benchmark, not a
# numerical match: the exact AdamW weight decay etc. differ, so accuracies only track loosely.

import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, disable_progress_bars
from datasets.utils.logging import set_verbosity_error
from torch.utils.data import DataLoader, Dataset, TensorDataset

disable_progress_bars()
set_verbosity_error()

BATCHSIZE = 128
NHIDDEN = 100
EPOCHS = 4
DEVICE = torch.device("cpu")


def make_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, NHIDDEN), nn.ReLU(),
        nn.Linear(NHIDDEN, NHIDDEN), nn.ReLU(),
        nn.Linear(NHIDDEN, 10),
    ).to(DEVICE)


class HFImageDataset(Dataset):
    """On-the-fly loading: index the Arrow-backed HF dataset and decode per item.

    Mirrors `mapobs(mnist_transform, ds)` in flux_mnist.jl. The dataset is memory-mapped and
    picklable, so with num_workers>0 each worker process re-opens it (data shared, not
    copied) — the same mechanism as the Julia `num_workers` path.
    """

    def __init__(self, hf_ds):
        self.ds = hf_ds.with_format("numpy")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        image = np.asarray(row["image"], dtype=np.float32) / 255.0  # (28, 28) in [0, 1]
        return torch.from_numpy(image), int(row["label"])


def materialize(hf_ds):
    """Decode the whole split into in-memory tensors up front (the `[:]` path in flux_mnist.jl)."""
    ds = hf_ds.with_format("numpy")
    col = ds["image"]
    images = col if isinstance(col, np.ndarray) else np.stack(col)
    images = torch.from_numpy(images.astype(np.float32) / 255.0)  # (N, 28, 28)
    labels = torch.from_numpy(np.asarray(ds["label"]).astype(np.int64))  # (N,)
    return TensorDataset(images, labels)


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


def train(num_workers=0, materialize_data=False):
    train_hf = load_dataset("ylecun/mnist", split="train")
    test_hf = load_dataset("ylecun/mnist", split="test")

    if materialize_data:
        train_ds, test_ds = materialize(train_hf), materialize(test_hf)
    else:
        train_ds, test_ds = HFImageDataset(train_hf), HFImageDataset(test_hf)

    # persistent_workers keeps the pool alive across epochs (closer to Julia's leased pool);
    # without it PyTorch respawns workers on every pass over the loader.
    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, **kw)
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, **kw)

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossfn = nn.CrossEntropyLoss()

    def report(epoch):
        tr_loss, tr_acc = loss_and_accuracy(train_loader, model)
        te_loss, te_acc = loss_and_accuracy(test_loader, model)
        print(f"(epoch = {epoch}, train_loss = {tr_loss:.3f}, train_acc = {tr_acc:.3f}, "
              f"test_loss = {te_loss:.3f}, test_acc = {te_acc:.3f})")

    report(0)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossfn(model(x), y)
            loss.backward()
            opt.step()
        report(epoch)


def timed(name, **kw):
    print(f"### {name}")
    t0 = time.perf_counter()
    train(**kw)
    print(f"  {time.perf_counter() - t0:.3f} seconds")


if __name__ == "__main__":
    timed("Serial", num_workers=0, materialize_data=False)
    timed("Serial Materialized", num_workers=0, materialize_data=True)
    timed("Parallel Materialized", num_workers=4, materialize_data=True)
    timed("Distributed", num_workers=4, materialize_data=False)
