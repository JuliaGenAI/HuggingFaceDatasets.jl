# AGENTS.md

Guidance for working in this repository.

## What this package is

HuggingFaceDatasets.jl is a (non-official) Julia wrapper around the Python
[`datasets`](https://huggingface.co/docs/datasets) library from Hugging Face.
It exposes Hugging Face datasets to Julia by wrapping the Python objects with
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) and adding Julia
conveniences (1-based indexing, copyless array conversion, a `"julia"` format
that lazily converts observations to Julia types).

## Layout

- `src/HuggingFaceDatasets.jl` — module entry point. Holds the lazily-imported
  Python module handles (`datasets`, `PIL`, `np`, `copy`, `pickle`), initialized in
  `__init__` via `PythonCall.pycopy!`. Python modules must be imported there,
  not at module top level (PythonCall restriction — see the comment in the file).
- `src/dataset.jl` — `Dataset`, the wrapper over `datasets.Dataset`. 1-based
  indexing, format/transform machinery (`with_format`, `set_format!`,
  `with_jltransform`, `set_jltransform!`, `reset_format!`).
- `src/datasetdict.jl` — `DatasetDict`, a dict of `Dataset`s.
- `src/iterabledataset.jl` — `IterableDataset`, the lazy streaming counterpart of
  `Dataset` (wraps `datasets.IterableDataset`). Consumed by `Base.iterate`, not
  indexing (no length / no random access); returned by
  `load_dataset(...; streaming=true)` with a `split`.
- `src/iterabledatasetdict.jl` — `IterableDatasetDict`, a dict of `IterableDataset`s
  (wraps `datasets.IterableDatasetDict`); returned by streaming `load_dataset`
  without a `split`.
- `src/column.jl` — `Column`, a lazy 1-based `AbstractVector` view over a single
  dataset column (wraps the `datasets.Column` returned by `dataset[name]` on
  datasets ≥ 4), converting elements with `py2jl` on access.
- `src/load_dataset.jl` — `load_dataset`, thin wrapper over
  `datasets.load_dataset` returning a `Dataset`/`DatasetDict` (or, with
  `streaming=true`, an `IterableDataset`/`IterableDatasetDict`).
- `src/toplevel.jl` — Julia wrappers for module-level `datasets` functions
  (`concatenate_datasets`, `interleave_datasets`, `load_from_disk`, and
  `from_csv`/`from_json`/`from_parquet`) that re-wrap results in the default
  `"julia"` format.
- `src/transforms.jl` — `py2jl` / `numpy2jl` / `jl2numpy` / `jl2py`. `py2jl`
  recursively converts Python containers, numpy arrays (copyless, zero-copy), and
  PIL images into Julia types; `jl2py` is the write-path dual. The `"julia"`
  format is numpy-backed, so numeric array columns decode to real N-D Julia arrays
  and image columns decode to raw numeric arrays (not `Colorant` colorviews).
- `src/features.jl` — `Py`-backed views over a dataset's schema: `Features` (an
  `AbstractDict` returned by `ds.features`), and the `ClassLabel`/`Value` leaves it
  wraps, each forwarding attribute/method access to Python (`cl.names`,
  `cl.int2str`, `v.dtype`). Handled at the access site (a `:features` branch in
  `Dataset`'s `getproperty`), never in the `py2jl` batch hot path. Also the Julian
  label-decoding helpers `class_names`/`int2str`/`str2int` (`(ds, col, …)`), and
  `jl2py` overloads so a Julia-built schema round-trips into a `features=` argument.
  Everything here is public but unexported; the Pythonic idioms are primary.
- `src/serialization.jl` — `Serialization.serialize`/`deserialize` for `Dataset`,
  so it can cross a process boundary (process-parallel data loaders). Never
  serializes the wrapped `Py` directly; instead uses `datasets`' own pickle
  (on-disk datasets pickle by reference to their mmapped Arrow `cache_files`;
  in-memory ones are materialized to a temp Arrow dir once, fingerprint-cached).
- `src/callable.jl` — small helper for method forwarding to the wrapped Python
  object. The `getobs`/`numobs` observation interface comes from `MLCore` (≥ 1.1);
  `getobs(::Py, ::Integer)` is provided by MLCore's `PythonCall` extension, so the
  package no longer carries its own (pirate) method for it.

## Python dependencies

Managed by [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) via
`CondaPkg.toml` (conda-forge channel): `datasets`, `numpy`, `pillow`.

- `datasets` is pinned to `>=4.0, <5`. Do **not** lower the floor below 4.0
  (the lazy `Column` column-access path depends on datasets ≥ 4).
- `python` is pinned to `<3.14` to avoid conda-forge's free-threaded (no-GIL)
  CPython, which deadlocks PythonCall while importing `datasets`' C extensions.

## Running tests

Use the Julia MCP server (`julia_eval`) rather than the `julia` CLI, with
`env_path` set to the repo root:

```julia
using Pkg; Pkg.test()
```

- Test deps live in `test/Project.toml` (`Test`, `ImageCore`, `PythonCall`);
  the root project declares `[workspace] projects = ["test", "docs"]`.
- `test/runtests.jl` runs `transforms.jl`, `dataset.jl`, `datasetdict.jl`,
  `iterabledataset.jl`, and `iterabledatasetdict.jl` always (the last two stay
  offline, building streams from in-memory data), and `no_ci.jl` (larger downloads:
  cifar10, beans, cppe-5) **only when `CI` is not `"true"`**. Set `ENV["CI"]="true"`
  to mimic CI and skip those.
- The first run downloads datasets and provisions the conda env, so it is slow.

## Changelog

PRs should update [`CHANGELOG.md`](CHANGELOG.md). It follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/).

- Any user-visible change (bug fix; added, changed, deprecated, or removed method or
  behavior) must add a bullet under the `## [Unreleased]` heading, in the matching
  category: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or `Security`.
  Purely internal changes (refactors, tests, CI, doc typos) don't need an entry.
- Write entries for users, in the present tense, describing the effect rather than the
  code change.
- On release: rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty
  `## [Unreleased]` above it, bump `version` in `Project.toml` per SemVer (new feature
  → minor, bug fix → patch; for `0.x`, breaking → minor), and update the compare links
  at the bottom of the file.
