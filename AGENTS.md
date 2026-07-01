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
  Python module handles (`datasets`, `PIL`, `np`, `copy`), initialized in
  `__init__` via `PythonCall.pycopy!`. Python modules must be imported there,
  not at module top level (PythonCall restriction — see the comment in the file).
- `src/dataset.jl` — `Dataset`, the wrapper over `datasets.Dataset`. 1-based
  indexing, format/transform machinery (`with_format`, `set_format!`,
  `with_jltransform`, `set_jltransform!`, `reset_format!`).
- `src/datasetdict.jl` — `DatasetDict`, a dict of `Dataset`s.
- `src/load_dataset.jl` — `load_dataset`, thin wrapper over
  `datasets.load_dataset` returning a `Dataset` or `DatasetDict`.
- `src/transforms.jl` — `py2jl` / `numpy2jl` / `jl2numpy`. `py2jl` recursively
  converts Python containers, numpy arrays (copyless via DLPack), and PIL images
  (to `RGB{N0f8}` / `Gray{N0f8}`) into Julia types. This is what the `"julia"`
  format applies.
- `src/callable.jl`, `src/observation.jl` — small helpers (method forwarding to
  the wrapped Python object; `MLUtils` `getobs`/`numobs` integration).

## Python dependencies

Managed by [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) via
`CondaPkg.toml` (conda-forge channel): `datasets`, `numpy`, `pillow`.

- `datasets` is pinned to `>=3.0, <4`. Do **not** lower the floor below 3.0.

## Running tests

Use the Julia MCP server (`julia_eval`) rather than the `julia` CLI, with
`env_path` set to the repo root:

```julia
using Pkg; Pkg.test()
```

- Test deps live in `test/Project.toml` (`Test`, `ImageCore`, `PythonCall`);
  the root project declares `[workspace] projects = ["test", "docs"]`.
- `test/runtests.jl` runs `dataset.jl` and `datasetdict.jl` always, and
  `no_ci.jl` (larger downloads: cifar10, beans, cppe-5) **only when `CI` is
  not `"true"`**. Set `ENV["CI"]="true"` to mimic CI and skip those.
- The first run downloads datasets and provisions the conda env, so it is slow.

## Changelog

Keep [`CHANGELOG.md`](CHANGELOG.md) up to date. It follows
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
