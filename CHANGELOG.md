# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `MLUtils.DataLoader(ds::Dataset; num_workers=N)` works out of the box for process-parallel
  loading, including over `mapobs`/`ObsView`-wrapped datasets. Building on the `Serialization`
  support added in 0.4.0, the GIL-safe serialization it relies on — running the dataset's
  `pickle` on the main task and loading this package on the workers — is handled upstream by
  MLUtils, so no wrapper type is exported here. An `@info` notes when an in-memory dataset is
  first materialized to a temporary Arrow file for this. Requires MLUtils ≥ 0.4.11.

## [0.4.1] - 2026-07-06

### Added
- Julia views over a dataset's schema. `ds.features` now returns a `Features` view (an
  `AbstractDict{String, Any}`) instead of a raw `Py` mapping: indexing a column yields a wrapped
  `ClassLabel`/`Value` leaf (other feature types stay raw `Py`), each forwarding attribute and
  method access to Python (`cl.names`, `cl.num_classes`, `cl.int2str(i)`, `cl.str2int(s)`,
  `v.dtype`). The views can be built from Julia (`ClassLabel(names=[…])`, `Value("int64")`,
  `Features(Dict(…))`) and handed back to Python via `jl2py` (e.g. a `features=` schema
  argument). Public but unexported Julian conveniences look up a column's `ClassLabel` in one
  call: `class_names(ds, col)`, `int2str(ds, col, i)`, `str2int(ds, col, s)` (label ids are
  0-based data, passed through with no index offset), plus `features(ds)` as the function form
  of `ds.features`. `Features`/`ClassLabel`/`Value` and these helpers are all public but not
  exported — the Pythonic idioms (`ds.features`, method chaining, `datasets.ClassLabel(…)`) are
  the primary interface.

### Changed
- Depend on `MLCore` (≥ 1.1) instead of `MLUtils` for the `getobs`/`numobs` observation
  interface, and drop the package's own `getobs(::Py, ::Integer)` method (type piracy on
  `PythonCall.Py`). `getobs` for Python observation containers is now provided by MLCore's
  `PythonCall` extension. `MLUtils.DataLoader` still works, since MLUtils re-exports the
  same `getobs`/`numobs` from MLCore.

## [0.4.0] - 2026-07-02

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
- `Serialization` support for `Dataset`: `Serialization.serialize`/`deserialize` now work,
  so a `Dataset` can cross a process boundary — the prerequisite for process-parallel data
  loaders (e.g. a `MLUtils.DataLoader(ds; num_workers=N)` that spreads `getobs` over worker
  processes, where separate interpreters/GILs let Python reads scale in a way the shared-GIL
  thread path cannot). It uses `datasets`' own pickle (the mechanism PyTorch relies on for
  `DataLoader(num_workers>0)`): an on-disk dataset pickles to a small reference to its
  memory-mapped Arrow `cache_files` and re-mmaps on the worker (data shared, not copied),
  while an in-memory dataset is materialized to a temp Arrow dir once (fingerprint-cached).
  Only the pickle bytes and the Julia transform cross the boundary — never a `Py`. Requires
  a Julia-native read format (the default `"julia"`), a serializable `jltransform`, and a
  shared filesystem across processes.
- Streaming support: `load_dataset(...; streaming=true)` now returns an exported
  `IterableDataset` (with a `split`) or `IterableDatasetDict` (without one) instead of leaking
  a raw `Py`. `IterableDataset` is the lazy counterpart of `Dataset` — it is consumed by
  iteration (`for obs in itds`, `collect`, `Iterators.take`), not indexing (it has no length or
  random access, which raise an explanatory `ArgumentError`), and defaults to the `"julia"`
  format so each yielded row converts to native Julia types. Its lazy transforms
  (`take`/`skip`/`shuffle(buffer_size=…)`/`map`/`filter`) are forwarded and re-wrapped, the
  julia-bridged `map`/`filter` overloads (and `ds.map`/`ds.filter`) work as for `Dataset`, and
  `with_format`/`set_format!`/`with_jltransform` mirror the `Dataset` API.
  `IterableDatasetDict` is the `AbstractDict{String, IterableDataset}` analogue of `DatasetDict`.
- `Dataset(table)` construction from any [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible
  source (`DataFrame`, `CSV.File`, row tables, …). The table is materialized with
  `Tables.columntable` and fed through the existing `Dataset(::NamedTuple)` column path,
  avoiding the pandas dependency of `datasets.Dataset.from_pandas` while covering many more
  sources.
- Top-level `datasets` functions are now exposed as Julia wrappers that re-wrap their
  results in the default `"julia"` format instead of returning a raw `Py`:
  `concatenate_datasets`, `interleave_datasets`, and `load_from_disk` (exported; the last
  closes the `save_to_disk`/load asymmetry and auto-detects `Dataset` vs `DatasetDict`), plus
  `from_csv`/`from_json`/`from_parquet` (public but not exported).
- Python-classmethod-style access via type-level `getproperty`: `Dataset.from_dict(...)`,
  `Dataset.from_csv`/`Dataset.from_json`/`Dataset.from_parquet`, and
  `Dataset.load_from_disk`/`DatasetDict.load_from_disk`, mirroring the Python API and routing
  to the wrappers above.
- `jl2py` on a `Dataset`/`DatasetDict`/`Column` unwraps to the underlying Python object, so
  the wrapper types can be passed directly as arguments to Python calls.
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
- The Python-style `getproperty` interface now bridges Julia values consistently:
  `ds.set_format(...)`/`ds.reset_format()` (and the `DatasetDict` forms) route through the
  julia format methods (so `ds.set_format("julia")` works and matches the `set_format!`/
  `reset_format!` functions), and `ds.map(f)`/`ds.filter(f)` bridge Julia values so that
  `ds.map(f) == map(f, ds)`. `map(f, dd)` was added for `DatasetDict`, and `filter(f, dd)`
  now filters *examples* like `dd.filter(f)` (deliberately overriding the `AbstractDict`
  split-filter).
- Renamed the internal field of `Column` from `pyobj` to `py`, matching `Dataset`/
  `DatasetDict`. Not part of the public API.
- `DatasetDict` now has a dedicated `text/plain` display mirroring the Python
  `datasets.DatasetDict` repr (nested `Dataset` summaries), instead of the generic
  `AbstractDict` multi-line display.
- `py2jl` now converts any `PIL.Image.Image` (BMP/GIF/TIFF/WebP and images produced by
  transforms), not only PNG/JPEG. Image modes other than RGB and grayscale (e.g. RGBA,
  CMYK, palette) return the raw array instead of raising an error.
- `py2jl` read-path robustness: non-numeric numpy arrays (strings, ragged/object,
  datetimes) fall back from DLPack to a nested-list conversion; numpy scalars
  (`np.str_`, `np.int64`, …) convert via `.item()`; 0-dimensional numpy arrays (which the
  numpy formatter produces when tensorizing a plain scalar cell) unwrap to native scalars
  rather than `fill(x)` 0-d Julia arrays; and read-only numpy buffers are copied before
  `from_dlpack`.
- `get(::DatasetDict, key, default)` now does a single Python round-trip (via
  `dict.get`) instead of a separate `haskey` + lookup.
- Renamed the internal field holding the wrapped Python object to `py` on both
  `Dataset` (was `pyds`) and `DatasetDict` (was `pyd`), for consistency. This field is
  not part of the public API (it is shadowed by `getproperty`), so the change is
  non-breaking for documented usage.

### Fixed
- Fixed a segfault triggered by REPL tab-completion on a `DatasetDict` (`d[<TAB>`), which
  called into libpython off the main task. `DatasetDict` split names are now cached, so
  `keys`/`length`/`haskey` are Python-free (also a small performance win).
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

[Unreleased]: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/releases/tag/v0.3.4
