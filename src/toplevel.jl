# Module-level wrappers for the top-level Python `datasets` surface that is not reachable as
# a method on a `Dataset`/`DatasetDict`. Each wrapper unwraps its `Dataset`/`DatasetDict`
# arguments to the underlying Python object (via the `jl2py` overloads in `transforms.jl`),
# calls the Python function, and re-wraps the result with `_wrap_toplevel` — the same
# one-rewrap boundary the `Dataset` methods already provide, so results come back with
# 1-based indexing, the `"julia"` format, and further method forwarding.

# Re-wrap a top-level Python result. A `datasets.Dataset`/`DatasetDict` comes back in the
# default `"julia"` format (like `load_dataset` and the `Dataset` constructors, so
# observations convert to native Julia types on access); anything else goes through the
# generic `py2jl`. Use the underlying `.py` / `set_format!(_, nothing)` for raw Python.
function _wrap_toplevel(x::Py)
    y = py2jl(x)
    return y isa Union{Dataset,DatasetDict} ? set_format!(y, "julia") : y
end

"""
    concatenate_datasets(dsets::AbstractVector; axis=0, kws...)
    concatenate_datasets(dsets::Dataset...; axis=0, kws...)

Concatenate several [`Dataset`](@ref)s into a single one, forwarding to
[`datasets.concatenate_datasets`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.concatenate_datasets).
Pass the datasets either as a vector or as individual arguments.

With `axis=0` (the default) the datasets are stacked row-wise (they must share the same
columns); with `axis=1` they are concatenated column-wise (they must have the same number of
rows). Extra keyword arguments (`info`, `split`, ...) are forwarded to Python. The result is
returned in the default `"julia"` format.

# Examples

```jldoctest
julia> a = Dataset((; label=[1, 2]));

julia> b = Dataset((; label=[3, 4, 5]));

julia> ds = concatenate_datasets(a, b)
Dataset({
    features: ['label'],
    num_rows: 5
})

julia> ds[:]["label"]
5-element Vector{Int64}:
 1
 2
 3
 4
 5
```
"""
function concatenate_datasets(dsets::AbstractVector; kws...)
    return _wrap_toplevel(datasets.concatenate_datasets(jl2py(dsets); kws...))
end

concatenate_datasets(dsets::Dataset...; kws...) = concatenate_datasets(collect(dsets); kws...)

"""
    interleave_datasets(dsets::AbstractVector; probabilities=nothing, seed=nothing, kws...)
    interleave_datasets(dsets::Dataset...; probabilities=nothing, seed=nothing, kws...)

Interleave several [`Dataset`](@ref)s into a single one by alternating between them,
forwarding to
[`datasets.interleave_datasets`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.interleave_datasets).
Pass the datasets either as a vector or as individual arguments.

Without `probabilities`, examples are taken from each dataset in round-robin order; with
`probabilities` (a vector summing to 1) each next example is sampled from a dataset according
to those weights (pass `seed` for reproducibility). Extra keyword arguments
(`stopping_strategy`, ...) are forwarded to Python. The result is returned in the default
`"julia"` format.

# Examples

```jldoctest
julia> a = Dataset((; label=[1, 1, 1]));

julia> b = Dataset((; label=[2, 2, 2]));

julia> ds = interleave_datasets(a, b);

julia> ds[:]["label"]
6-element Vector{Int64}:
 1
 2
 1
 2
 1
 2
```
"""
function interleave_datasets(dsets::AbstractVector; kws...)
    return _wrap_toplevel(datasets.interleave_datasets(jl2py(dsets); kws...))
end

interleave_datasets(dsets::Dataset...; kws...) = interleave_datasets(collect(dsets); kws...)

"""
    load_from_disk(path; kws...)

Load a [`Dataset`](@ref) or [`DatasetDict`](@ref) previously written with `save_to_disk`,
forwarding to
[`datasets.load_from_disk`](https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_from_disk).
This is the read side of the `ds.save_to_disk(path)` method, closing the save/load asymmetry.

The result is returned in the default `"julia"` format. Extra keyword arguments
(`keep_in_memory`, `storage_options`, ...) are forwarded to Python. The Python classmethods
are also available under their names: `Dataset.load_from_disk(path)` and
`DatasetDict.load_from_disk(path)` load specifically a `Dataset` / `DatasetDict`, whereas this
top-level `load_from_disk` auto-detects which one was saved.

# Examples

```julia
julia> ds = Dataset((; label=[5, 0, 4]));

julia> ds.save_to_disk("mydataset");

julia> load_from_disk("mydataset")
Dataset({
    features: ['label'],
    num_rows: 3
})
```
"""
load_from_disk(path::AbstractString; kws...) =
    _wrap_toplevel(datasets.load_from_disk(path; kws...))

# The file-based construction family. These mirror the Python classmethods
# `datasets.Dataset.from_{csv,json,parquet}` and give a no-pandas ingestion path alongside
# the in-memory `Dataset(::AbstractDict)` / Tables.jl constructors. They are public but not
# exported (call them as `HuggingFaceDatasets.from_csv(...)`), keeping the Python name while
# leaving the exported namespace to the `Dataset` constructor. `path_or_paths` may be a file
# path, a vector of paths, or a mapping of split name to path; keyword arguments (`features`,
# `cache_dir`, `keep_in_memory`, ...) are forwarded to Python, and the result comes back in
# the default `"julia"` format. They are also reachable under the Python classmethod name via
# the type-level `getproperty` on `Dataset` (`Dataset.from_csv(path)` etc., see `dataset.jl`).

"""
    from_csv(path_or_paths; kws...)
    Dataset.from_csv(path_or_paths; kws...)

Build a [`Dataset`](@ref) from a CSV file (or files), forwarding to
`datasets.Dataset.from_csv`. Reachable both as the (public, not exported)
`HuggingFaceDatasets.from_csv` and under the Python classmethod name `Dataset.from_csv`.
See also [`from_json`](@ref) and [`from_parquet`](@ref).
"""
from_csv(path_or_paths; kws...) =
    _wrap_toplevel(datasets.Dataset.from_csv(jl2py(path_or_paths); kws...))

"""
    from_json(path_or_paths; kws...)
    Dataset.from_json(path_or_paths; kws...)

Build a [`Dataset`](@ref) from a JSON / JSON Lines file (or files), forwarding to
`datasets.Dataset.from_json`. Reachable both as the (public, not exported)
`HuggingFaceDatasets.from_json` and under the Python classmethod name `Dataset.from_json`.
See also [`from_csv`](@ref) and [`from_parquet`](@ref).
"""
from_json(path_or_paths; kws...) =
    _wrap_toplevel(datasets.Dataset.from_json(jl2py(path_or_paths); kws...))

"""
    from_parquet(path_or_paths; kws...)
    Dataset.from_parquet(path_or_paths; kws...)

Build a [`Dataset`](@ref) from a Parquet file (or files), forwarding to
`datasets.Dataset.from_parquet`. Reachable both as the (public, not exported)
`HuggingFaceDatasets.from_parquet` and under the Python classmethod name `Dataset.from_parquet`.
See also [`from_csv`](@ref) and [`from_json`](@ref).
"""
from_parquet(path_or_paths; kws...) =
    _wrap_toplevel(datasets.Dataset.from_parquet(jl2py(path_or_paths); kws...))
