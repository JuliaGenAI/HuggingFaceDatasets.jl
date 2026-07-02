# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This is a breaking release (0.3.x → 0.4.0). The headline change is that datasets are
now returned in the `"julia"` format by default, so observations come back as native
Julia values instead of raw Python objects. See **Breaking** below before upgrading.

### Breaking
- **`"julia"` is now the default format.** `load_dataset`, `DatasetDict`, and the new
  `Dataset(::AbstractDict/::NamedTuple)` constructors return datasets in the `"julia"`
  format, so indexing an observation (`ds[i]`), a column (`ds["label"]`), and iteration
  all convert to native Julia types instead of handing back raw `Py` objects. Wrapping a
  raw `datasets.Dataset` directly (`Dataset(pyds)`) stays format-neutral. Opt back out
  with `set_format!(ds, nothing)` (or `with_format(ds, nothing)`) for raw Python
  observations.
- **The `"julia"` format is now numpy-backed.** It sets the underlying Python format to
  `numpy`, so numeric array columns decode to real N-D Julia arrays and a range index
  stacks rows into a `(dims…, N)` tensor with the observation axis last (the MLUtils
  convention). Consequently, **image columns now decode to raw numeric arrays** rather
  than `Gray`/`RGB` colorviews — update downstream code that expected `Colorant` element
  types.
- Bumped the `datasets` Python dependency to `>=4.0, <5` (previously `<4`). Environments
  pinned to `datasets` 3.x are no longer supported.
- `reset_format!` now restores the default `"julia"` format instead of stripping all
  formatting; use `set_format!(ds, nothing)` to strip to raw Python observations.
- `py2jl` on a `datasets.Column` (and hence string indexing of a julia-formatted
  `Dataset`, e.g. `ds["label"]`) now returns a lazy `Column` view instead of eagerly
  materializing the whole column. The result still behaves like a vector (indexing,
  slicing, iteration, broadcasting, comparison); call `collect` to get a plain `Vector`.
- Invalid arguments now raise `ArgumentError`, and out-of-range indices raise
  `BoundsError`, instead of `AssertionError` — update any code catching `AssertionError`.

### Added
- `Dataset(::AbstractDict)` and `Dataset(::NamedTuple)` constructors that build a dataset
  from in-memory Julia column vectors (via `datasets.Dataset.from_dict`), replacing the
  long-standing commented-out stub. Scalar columns and array-valued columns (given as a
  vector-of-arrays or a single stacked `(dims…, N)` array, last axis = observations) are
  supported; elements go through the `jl2py`/`jl2numpy` write path.
- `DatasetDict` constructors from in-memory Julia data — a mapping of split names to
  `Dataset`s, given as an `AbstractDict{<:AbstractString, Dataset}` or as `name => ds`
  pairs (`DatasetDict("train" => train, "test" => test)`). Each split is shallow-copied so
  the source `Dataset`s are not mutated, and inherits its source `Dataset`'s own transform
  (so a dict built from `Dataset((; ...))`s is in the `"julia"` format; change it afterwards
  with `set_jltransform!`/`set_format!`). Mirrors the `Dataset(::AbstractDict/::NamedTuple)`
  constructors and replaces the `DatasetDict(datasets.DatasetDict(pydict(…)))` boilerplate.
- Per-split julia transforms on `DatasetDict`. Indexing a split hands back a `Dataset`
  carrying that split's own transform, and `set_jltransform!`/`with_jltransform` accept an
  `AbstractDict` mapping split names to transforms (a single callable still applies to all
  splits; `set_format!`/`reset_format!` remain split-wide). `merge`/`filter` and the
  constructors preserve each split's transform.
- Julia-friendly `map(f, ds::Dataset)` and `filter(f, ds::Dataset)` overloads that
  `py2jl`-convert each example/batch before `f` sees it and convert `f`'s Julia return
  back to Python, while still getting `datasets`' batching, caching, and multiprocessing.
  Keyword args are forwarded and the parent's julia format/transform is preserved.
  `ds.map(...)` still reaches the raw Python-callback method.
- `jl2py` is now public and exported — the write-path dual of `py2jl`.
- `DatasetDict <: AbstractDict{String, Dataset}` with the full dictionary interface
  (`keys`, `values`, `pairs`, `haskey`, `get`, `iterate`), so splits can be accessed
  idiomatically. `merge` and `filter` on a `DatasetDict` now return a `DatasetDict`.
- `Column`, a lazy 1-based `AbstractVector` view over a single dataset column. It
  wraps the python `datasets.Column` returned by `dataset[column_name]` (datasets ≥ 4)
  and converts elements with `py2jl` on access; `collect(col)` materializes it.
- `Base.firstindex`/`Base.lastindex` for `Dataset`, so `ds[begin]` and `ds[end]` work.
- `Base.iterate` for `Dataset`; it iterates over observations, enabling
  `for obs in ds`, `collect(ds)`, and comprehensions.
- `reset_format!(::DatasetDict)` and a single-argument `set_format!(::DatasetDict)`,
  mirroring the `Dataset` methods.
- `Base.copy` for `Dataset`/`DatasetDict` (Python `copy.copy`): the copy shares the
  underlying Arrow data but has an independent format/transform. `with_format`/
  `with_jltransform` now use this shallow copy instead of `deepcopy`, so flipping a
  format flag no longer duplicates the dataset.

### Changed
- `DatasetDict` now has a dedicated `text/plain` display mirroring the Python
  `datasets.DatasetDict` repr (nested `Dataset` summaries), instead of the generic
  `AbstractDict` multi-line display.
- `py2jl` now converts any `PIL.Image.Image` (BMP/GIF/TIFF/WebP and images produced by
  transforms), not only PNG/JPEG. Image modes other than RGB and grayscale (e.g. RGBA,
  CMYK, palette) return the raw array instead of raising an error.
- `py2jl` read-path robustness: non-numeric numpy arrays (strings, ragged/object,
  datetimes) fall back from DLPack to a nested-list conversion; numpy scalars
  (`np.str_`, `np.int64`, …) convert via `.item()`; and read-only numpy buffers are
  copied before `from_dlpack`.
- `get(::DatasetDict, key, default)` now does a single Python round-trip (via
  `dict.get`) instead of a separate `haskey` + lookup.
- Renamed the internal field holding the wrapped Python object to `py` on both
  `Dataset` (was `pyds`) and `DatasetDict` (was `pyd`), for consistency. This field is
  not part of the public API (it is shadowed by `getproperty`), so the change is
  non-breaking for documented usage.

### Fixed
- Keyword arguments are correctly forwarded to wrapped Python methods (e.g.
  `ds.train_test_split(test_size=0.2)`, `ds.shuffle(seed=…)`, `ds.map(batched=true)`).
  They were previously passed as positional arguments and rejected by Python.
- `jl2numpy` works again with numpy ≥ 2.1, which had broken the DLPack export path. It
  now shares memory through the buffer protocol; the returned numpy array is writable,
  and mutations propagate in both directions.
- `py2jl` on a Python tuple now returns a proper Julia tuple of converted elements. It
  previously returned a 1-tuple wrapping an unevaluated generator.
- The `"julia"` format (and any custom `jltransform`) now survives forwarded methods that
  return a new dataset (`shuffle`, `select`, `sort`, `filter`, `map`, `train_test_split`,
  …). The parent's transform is propagated onto the result, mirroring how `datasets`
  propagates Python-side formats; previously it was dropped and the result came back as a
  raw `Py`.
- `copy(::DatasetDict)` now shallow-copies each split, so a format change on the copy no
  longer leaks to the original (copy-on-write was broken for `DatasetDict`).
- `deepcopy` now works when a dataset is nested inside another object being copied. It
  overrides `deepcopy_internal` (the correct extension point) rather than `deepcopy`, so
  the wrapped Python object is properly duplicated instead of aliased.

## [0.3.4]

Baseline. Changes up to and including this release are recorded in the
[git history](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/commits/main) and the
[GitHub releases](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/releases).

[Unreleased]: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/releases/tag/v0.3.4
