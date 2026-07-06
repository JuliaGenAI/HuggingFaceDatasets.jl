# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "datasets", "numpy", "pillow", "tqdm"]
# ///
#
# Idiomatic HuggingFace `datasets` + PyTorch MNIST training — written the way a practitioner
# typically would, rather than line-for-line equivalent to the Flux `flux_mnist.jl` (see
# `pytorch_mnist.py` for that). Run with:  uv run examples/flux_mnist/pytorch_mnist_idiomatic.py
#
# Idioms shown:
#   * `load_dataset` -> DatasetDict, then `ds["train"]` / `ds["test"]`
#   * a torchvision transform applied lazily via `Dataset.with_transform` (the canonical image
#     pipeline) — plus the `.map(batched=True)` + torch-format cached alternative
#   * a `collate_fn` feeding a standard `DataLoader(num_workers=...)`
#   * a plain `nn.Module`, `AdamW`, and a `tqdm` training loop
#
# The model, batch size and epochs match the Flux run so data-loading timings are comparable.
# Kept on CPU for that comparison; a typical script would auto-select the accelerator (see below).

import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, disable_progress_bars
from datasets.utils.logging import set_verbosity_error
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

disable_progress_bars()
set_verbosity_error()

BATCHSIZE = 128
EPOCHS = 4
# A typical script auto-selects the accelerator:
#   DEVICE = torch.device("cuda" if torch.cuda.is_available()
#                         else "mps" if torch.backends.mps.is_available() else "cpu")
# Pinned to CPU here so the timings line up with the Flux example.
DEVICE = torch.device("cpu")

# ToTensor scales uint8 [0, 255] -> float [0, 1] and lays out (C, H, W); Normalize applies the
# standard MNIST mean/std. This is the usual torchvision image pipeline.
IMAGE_TF = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


class MLP(nn.Module):
    def __init__(self, nhidden=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, nhidden), nn.ReLU(),
            nn.Linear(nhidden, nhidden), nn.ReLU(),
            nn.Linear(nhidden, 10),
        )

    def forward(self, x):
        return self.net(x)


def lazy_transform(batch):
    """Decode + transform on access (registered with `with_transform`)."""
    batch["pixel_values"] = [IMAGE_TF(img) for img in batch["image"]]
    return batch


def collate(examples):
    x = torch.stack([e["pixel_values"] for e in examples])
    y = torch.tensor([e["label"] for e in examples])
    return x, y


def cached_preprocess(batch):
    """Same transform, but returning arrays so `datasets` can cache them to Arrow."""
    batch["pixel_values"] = [IMAGE_TF(img).numpy() for img in batch["image"]]
    return batch


def make_loaders(num_workers, cached):
    ds = load_dataset("ylecun/mnist")
    if cached:
        # Preprocess once; `datasets` caches the result to Arrow and hands back torch tensors
        # on access, so the default collate works and no per-item Python decode is needed.
        ds = ds.map(cached_preprocess, batched=True, remove_columns=["image"])
        ds = ds.with_format("torch")
        train_split, test_split, collate_fn = ds["train"], ds["test"], None
    else:
        train_split = ds["train"].with_transform(lazy_transform)
        test_split = ds["test"].with_transform(lazy_transform)
        collate_fn = collate

    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0, collate_fn=collate_fn)
    train_loader = DataLoader(train_split, batch_size=BATCHSIZE, shuffle=True, **kw)
    test_loader = DataLoader(test_split, batch_size=BATCHSIZE, **kw)
    return train_loader, test_loader


def _xy(batch):
    # lazy path yields (x, y) tuples; cached torch-format path yields column dicts.
    if isinstance(batch, dict):
        return batch["pixel_values"], batch["label"]
    return batch


@torch.no_grad()
def evaluate(loader, model, lossfn):
    model.eval()
    total_loss, correct, num = 0.0, 0, 0
    for batch in loader:
        x, y = _xy(batch)
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += lossfn(logits, y).item() * y.shape[0]
        correct += (logits.argmax(1) == y).sum().item()
        num += y.shape[0]
    return total_loss / num, correct / num


def train(num_workers=0, cached=False):
    train_loader, test_loader = make_loaders(num_workers, cached)
    model = MLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossfn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}", leave=False):
            x, y = _xy(batch)
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossfn(model(x), y)
            loss.backward()
            opt.step()
        tr_loss, tr_acc = evaluate(train_loader, model, lossfn)
        te_loss, te_acc = evaluate(test_loader, model, lossfn)
        print(f"(epoch = {epoch}, train_loss = {tr_loss:.3f}, train_acc = {tr_acc:.3f}, "
              f"test_loss = {te_loss:.3f}, test_acc = {te_acc:.3f})")


def timed(name, **kw):
    print(f"### {name}")
    t0 = time.perf_counter()
    train(**kw)
    print(f"  {time.perf_counter() - t0:.3f} seconds")


if __name__ == "__main__":
    timed("Lazy with_transform, num_workers=0", cached=False, num_workers=0)
    timed("Lazy with_transform, num_workers=4", cached=False, num_workers=4)
    timed("Cached .map (torch format), num_workers=0", cached=True, num_workers=0)
    timed("Cached .map (torch format), num_workers=4", cached=True, num_workers=4)
