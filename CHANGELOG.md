# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `Base.firstindex`/`Base.lastindex` for `Dataset`, so `ds[begin]` and `ds[end]` work.
- `Base.iterate` for `Dataset`; it iterates over observations, enabling
  `for obs in ds`, `collect(ds)`, and comprehensions.
- `reset_format!(::DatasetDict)` and a single-argument `set_format!(::DatasetDict)`,
  mirroring the `Dataset` methods.

### Changed
- `DatasetDict` now has a dedicated `text/plain` display mirroring the Python
  `datasets.DatasetDict` repr (nested `Dataset` summaries), instead of the generic
  `AbstractDict` multi-line display.
- `py2jl` now converts any `PIL.Image.Image` (BMP/GIF/TIFF/WebP and images produced by
  transforms), not only PNG/JPEG. Image modes other than RGB and grayscale (e.g. RGBA,
  CMYK, palette) return the raw array instead of raising an error.
- Invalid arguments now raise `ArgumentError`, and out-of-range indices raise
  `BoundsError`, instead of `AssertionError`.
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

## [0.3.4]

Baseline. Changes up to and including this release are recorded in the
[git history](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/commits/main) and the
[GitHub releases](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/releases).

[Unreleased]: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/releases/tag/v0.3.4
