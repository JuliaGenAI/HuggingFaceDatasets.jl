# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "datasets", "numpy", "pillow"]
# ///
#
# PyTorch counterpart of flux_cifar10.jl (Flux + HuggingFaceDatasets), for a data-loading timing
# comparison on the same task: a small VGG-style CNN on CIFAR-10 pulled from the same HuggingFace
# `datasets` Arrow dataset, trained on the GPU with the standard crop+flip augmentation. Run with:
#   uv run perf/flux_cifar10/pytorch_cifar10.py
#
# The configs mirror flux_cifar10.jl. The on-the-fly path uses the idiomatic HF pattern
# (`Dataset.with_transform` + the standard torchvision crop→flip→ToTensor→Normalize order); the
# materialized path decodes the split into normalized CHW tensors once and then augments per item.
# One caveat: PyTorch's only DataLoader parallelism is multiprocess (`num_workers`), so the
# "Parallel Materialized" row uses worker *processes* over in-memory tensors — Julia's `parallel=true`
# there uses *threads* (cheap, shared memory), which PyTorch cannot do because of the GIL. This is a
# timing benchmark, not a numerical match: exact init/weight-decay etc. differ, so accuracies only
# track loosely.

import functools
import os
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, disable_progress_bars
from datasets.utils.logging import set_verbosity_error
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

disable_progress_bars()
set_verbosity_error()

BATCHSIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard CIFAR-10 per-channel mean/std (same values as flux_cifar10.jl).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# On-the-fly path — the usual torchvision order: augment the PIL image, then tensor + normalize.
TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     # zero-pad by 4, take a random 32x32 crop
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                    # PIL HWC uint8 -> CHW float in [0, 1]
    transforms.Normalize(MEAN, STD),
])
TEST_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

# Materialized path — tensors are already decoded+normalized CHW floats, so only the spatial
# augmentation remains (crop pads with zeros in normalized space, matching Julia's materialized path).
MAT_TRAIN_TF = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])


def make_model():
    """A small VGG-style CNN: three (conv-conv-pool) blocks, then a classifier. Mirrors flux_cifar10.jl."""
    def block(ci, co):
        return [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
    return nn.Sequential(
        *block(3, 64), *block(64, 64), nn.MaxPool2d(2),
        *block(64, 128), *block(128, 128), nn.MaxPool2d(2),
        *block(128, 256), *block(256, 256), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 10),
    ).to(DEVICE)


def apply_transform(batch, tf):
    """Decode + transform on access (registered with `with_transform`); picklable to workers."""
    batch["pixel_values"] = [tf(img) for img in batch["img"]]
    return batch


def collate(examples):
    x = torch.stack([e["pixel_values"] for e in examples])
    y = torch.tensor([e["label"] for e in examples])
    return x, y


class TensorAugDataset(Dataset):
    """Materialized path: hold decoded+normalized CHW float tensors in memory, augment per item."""

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


def materialize(hf_ds):
    """Decode the whole split into in-memory normalized tensors up front (the `[:]` path in Julia)."""
    ds = hf_ds.with_format("numpy")
    col = ds["img"]
    imgs = col if isinstance(col, np.ndarray) else np.stack(col)   # (N, H, W, C) uint8
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().float().div_(255.0)  # (N, C, H, W)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std
    labels = torch.from_numpy(np.asarray(ds["label"])).long()
    return imgs, labels


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


def make_loaders(num_workers, materialize_data):
    if materialize_data:
        train_hf = load_dataset("uoft-cs/cifar10", split="train")
        test_hf = load_dataset("uoft-cs/cifar10", split="test")
        train_ds = TensorAugDataset(*materialize(train_hf), MAT_TRAIN_TF)
        test_ds = TensorDataset(*materialize(test_hf))
        collate_fn = None
    else:
        ds = load_dataset("uoft-cs/cifar10")
        train_ds = ds["train"].with_transform(functools.partial(apply_transform, tf=TRAIN_TF))
        test_ds = ds["test"].with_transform(functools.partial(apply_transform, tf=TEST_TF))
        collate_fn = collate

    # persistent_workers keeps the pool alive across epochs (closer to Julia's leased pool) so the
    # discarded warm-up epoch below pays the worker-spawn cost once, out of the timed region.
    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0,
              pin_memory=(DEVICE.type == "cuda"), collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, **kw)
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, **kw)
    return train_loader, test_loader


def train(num_workers=0, materialize_data=False, verbose=False, loader_only=False):
    """Return wall-clock seconds of EPOCHS timed epochs; one warm-up epoch runs first and is discarded
    (spawns the persistent workers, warms cuDNN), so worker-spawn cost stays out of the timing.
    `verbose=True` is the DEMO path: report accuracy per epoch instead, no warm-up, no timing."""
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
    dev = "GPU" if DEVICE.type == "cuda" else "CPU"
    print(f"### DEMO — CNN learning CIFAR-10 on {dev} (accuracy per epoch)")
    train(num_workers=0, materialize_data=True, verbose=True)

    # Each config runs one warm-up epoch (discarded) before its timed epochs, so worker-spawn and
    # cuDNN autotune stay out of the numbers — matching the Julia script's warm-up discipline.
    print("\n#### FULL TRAINING — model + data loading ###########")
    timed("Serial", num_workers=0, materialize_data=False)
    timed("Serial Materialized", num_workers=0, materialize_data=True)
    timed("Parallel Materialized", num_workers=4, materialize_data=True)
    timed("Distributed (2 workers)", num_workers=2, materialize_data=False)
    timed("Distributed (4 workers)", num_workers=4, materialize_data=False)
    timed("Distributed (8 workers)", num_workers=8, materialize_data=False)
    print("#### END FULL TRAINING ###########")

    # Same configs, but iterating the loader with no model — the pure data-loading cost. Comparing
    # against the full-training numbers shows how much of each config is loading vs. GPU compute.
    print(f"\n#### DATA-LOADING ONLY — no model, same {EPOCHS} epochs ###########")
    timed("Serial", num_workers=0, materialize_data=False, loader_only=True)
    timed("Serial Materialized", num_workers=0, materialize_data=True, loader_only=True)
    timed("Parallel Materialized", num_workers=4, materialize_data=True, loader_only=True)
    timed("Distributed (2 workers)", num_workers=2, materialize_data=False, loader_only=True)
    timed("Distributed (4 workers)", num_workers=4, materialize_data=False, loader_only=True)
    timed("Distributed (8 workers)", num_workers=8, materialize_data=False, loader_only=True)
    print("#### END DATA-LOADING ONLY ###########")
