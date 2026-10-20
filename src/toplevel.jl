# Module-level wrappers for the top-level Python `datasets` surface that is not reachable as
# a method on a `Dataset`/`DatasetDict`. Each wrapper calls the Python function (wrapper types
# among the arguments reach Python as their underlying objects, see `HuggingFaceDatasets.jl`)
# and re-wraps the result with `_wrap_toplevel` — the same one-rewrap boundary the `Dataset`
# methods already provide, so results come back with 1-based indexing, the `"julia"` format,
# and further method forwarding.

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
    return _wrap_toplevel(datasets.concatenate_datasets(pylist(dsets); kws...))
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
    return _wrap_toplevel(datasets.interleave_datasets(pylist(dsets); kws...))
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

"""
    from_generator(source; batched = false, fingerprint = nothing, features = nothing,
                   split = nothing, cache_dir = nothing, kws...)
    Dataset.from_generator(source; kws...)
    Dataset(stream::IterableDataset; kws...)

Build a [`Dataset`](@ref) from the examples of `source`, forwarding to
[`datasets.Dataset.from_generator`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.from_generator).
The examples are written to an Arrow file in the cache, which the returned dataset
memory-maps. `source` can be

- an [`IterableDataset`](@ref), e.g. a stream from `load_dataset(...; streaming=true)` with
  `map`/`filter` steps, iterated in this process. This keeps only the result of a streamed
  pipeline, e.g. the token ids and not the text. The stream's `features` and `split` carry over.
  `Dataset(stream; kws...)` is the same.
- a Julia iterable of examples (`AbstractDict`s or `NamedTuple`s mapping column names to
  values), converted with [`jl2py`](@ref), or a function of no arguments returning one. With
  `batched = true`, the iterable yields batches instead: `AbstractDict`s or `NamedTuple`s mapping
  column names to columns, which are vectors or arrays whose last axis indexes the observations,
  as for [`Dataset(::AbstractDict)`](@ref). That is faster, since each column is converted at
  once.
- a Python generator function, handed to `datasets` as is, along with the other keyword
  arguments (`gen_kwargs`, `num_proc`, ...).

`features` sets the schema, otherwise inferred from the examples, and `split` the split name
(`"train"` by default). Other keyword arguments (`keep_in_memory`, ...) are forwarded to Python.
The result is returned in the default `"julia"` format.

!!! note "Caching"
    The dataset is cached under a fingerprint of `source`. A later call with the same
    fingerprint, `features` and `split`, also in a later session, loads it from the cache
    without iterating `source`. For a Julia source the fingerprint hashes what the source is and
    does, as for the function of [`map`](@ref): its values (the examples of a `Vector`, ...), the
    code of its functions and of the functions they use from your own modules, the values they
    capture and the globals they read, and `batched`. Files or other inputs from outside Julia,
    e.g. the lines of a file the source reads, are not tracked.

    An `IterableDataset`, or a source with something that can't be hashed, has no fingerprint:
    its dataset is rebuilt on every call, in a temporary directory deleted at the end of the
    session (or in `cache_dir`, if given). Pass `fingerprint`, a string naming the examples, to
    cache them under it instead, and change it when the examples change. As in Python, the cache
    is then named by `fingerprint` alone, so `datasets.Dataset.from_generator` with the same
    `fingerprint` and `cache_dir` shares it, as long as it has the same `split` (Python's default
    is `"train"`, while a stream's carries over): a call with another `split` throws.

!!! warning "No multiprocessing (`num_proc`)"
    The examples of a Julia source or of an `IterableDataset` are generated in this process, so
    `num_proc > 1` throws an `ArgumentError`. To parallelize a stream's Julia `map`, use
    `batched=true` and parallelize over each batch in the function, e.g. with `Threads.@threads`.

See also [`load_dataset`](@ref) and [`IterableDataset`](@ref).

# Examples

```julia
julia> ds = Dataset.from_generator((; x = i, y = Float32(i) / 2) for i in 1:3)
Dataset({
    features: ['x', 'y'],
    num_rows: 3
})

julia> ds[:]["y"]
3-element Vector{Float32}:
 0.5
 1.0
 1.5

julia> ds = Dataset.from_generator(; batched = true) do   # 2 batches of 3 rows
           (Dict("x" => reshape(Int32.(1:6) .+ Int32(6k), 2, 3)) for k in 0:1)
       end;

julia> ds[:]["x"]
2×6 Matrix{Int32}:
 1  3  5  7   9  11
 2  4  6  8  10  12
```

Tokenize a streamed corpus into blocks of 513 token ids, and keep only those in the cache:

```julia
julia> stream = load_dataset("roneneldan/TinyStories"; split = "validation", streaming = true);

julia> blocks = map(stream; batched = true, remove_columns = ["text"]) do b
           ids = tokenize(b["text"])                             # a Vector{UInt16}
           Dict("ids" => reshape(ids[1:end ÷ 513 * 513], 513, :))
       end;

julia> ds = Dataset(blocks; fingerprint = "tinystories-validation-gpt2-513");

julia> ds[1:32]["ids"]   # a 513×32 Matrix{UInt16}
```
"""
function from_generator(source; batched::Bool = false, kws...)
    examples, convert = batched ? (pyhfds.batch_rows, _batch_to_py) : (pyhfds.examples, _example_to_py)
    generator = pyfunc(() -> examples(_pynext(convert, source isa Function ? source() : source)))
    return _from_generator(generator, source, (; batched); kws...)
end

function from_generator(stream::IterableDataset; features = nothing, split = nothing, kws...)
    py = getfield(stream, :py)
    features = something(features, py.features)
    split = something(split, pyis(py.split, pybuiltins.None) ? datasets.Split.TRAIN : py.split)
    # Unformatted: Python values, which `features` encodes as they are in the stream
    return _from_generator(py.with_format(nothing).__iter__, stream, (;); features, split, kws...)
end

function from_generator(source::Py; kws...)
    pyisinstance(source, datasets.IterableDataset) &&
        return from_generator(IterableDataset(source); kws...)
    return _generate(source; kws...)
end

Dataset(stream::IterableDataset; kws...) = from_generator(stream; kws...)

# A given `fingerprint` goes to `datasets`' own keyword, which names the cache by it alone, so that
# Python's `from_generator` with the same `fingerprint` and `cache_dir` shares the cache. Otherwise
# `datasets` names it by hashing the generator function, which pickles as our content hash in a
# `JuliaFunction`, along with `features` and `split`.
function _from_generator(generator::Py, source, hashed_kws; fingerprint = nothing,
                         cache_dir = nothing, num_proc = nothing, kws...)
    if num_proc isa Integer && num_proc > 1
        throw(ArgumentError(
            "`from_generator` does not support `num_proc=$num_proc` for a Julia source or an " *
            "`IterableDataset`: it generates their examples in this process, since `datasets`' " *
            "worker processes cannot call back into Julia. Omit `num_proc`; to parallelize a " *
            "Julia function, have it work on batches with several threads, e.g. with " *
            "`Threads.@threads`."))
    end
    fingerprint === nothing || return _generate(generator; fingerprint, cache_dir, kws...)
    fp = _content_hash(source, hashed_kws)
    if !(fp isa AbstractString)
        @info "`from_generator` can't fingerprint its source ($(fp.msg)), so the dataset is \
               rebuilt on every call, in a temporary directory unless `cache_dir` is given. Pass \
               `fingerprint` to cache it." maxlog=1
        fp = _random_fingerprint()
        cache_dir = something(cache_dir, datasets.fingerprint.get_temporary_cache_files_directory())
    end
    return _generate(pyhfds.JuliaFunction(generator, fp); cache_dir, kws...)
end

# `datasets` raises an error of the generator as the cause of a `DatasetGenerationError`, which
# doesn't say what it is, so throw the cause instead.
function _generate(generator::Py; kws...)
    py = try
        datasets.Dataset.from_generator(generator; kws...)
    catch e
        (e isa PyException && pyisinstance(e.v, datasets.exceptions.DatasetGenerationError) &&
         !pyis(e.v.__cause__, pybuiltins.None)) || rethrow()
        throw(PyException(e.v.__cause__))
    end
    return _wrap_toplevel(py)
end

# A Python function returning the next element of `itr`, converted by `convert`, and
# `pyhfds.END` past the last one.
function _pynext(convert, itr)
    started = false
    state = nothing
    return pyfunc() do
        r = started ? iterate(itr, state) : iterate(itr)
        started = true
        r === nothing && return pyhfds.END
        x, state = r
        return convert(x)
    end
end

_example_to_py(x::Union{AbstractDict,NamedTuple,Py}) = jl2py(x)
_example_to_py(x) = throw(ArgumentError(
    "an example must be an `AbstractDict` or a `NamedTuple` of column values, got $(typeof(x))"))

# Each column at once: a stacked array becomes one numpy array, its observations along the first
# axis, which `pyhfds.batch_rows` splits.
function _batch_to_py(batch::Union{AbstractDict,NamedTuple})
    py = pydict()
    for (k, v) in pairs(batch)
        v isa AbstractArray || throw(ArgumentError(
            "column \"$k\" of a batch must be an `AbstractArray` of observations, got $(typeof(v))"))
        py[string(k)] = jl2py(v)
    end
    return py
end
_batch_to_py(x::Py) = x
_batch_to_py(x) = throw(ArgumentError(
    "with `batched = true`, each element must be an `AbstractDict` or a `NamedTuple` of " *
    "columns, got $(typeof(x))"))
