"""
    DatasetDict(splits::AbstractDict{<:AbstractString, Dataset})
    DatasetDict(splits::Pair{<:AbstractString, Dataset}...)

A `DatasetDict` is a dictionary of `Dataset`s.
It is a wrapper around a `datasets.DatasetDict` object.

A julia transform is stored **per split**: indexing a split (`dd["train"]`) hands back a
[`Dataset`](@ref) carrying that split's transform. The [`py2jl`](@ref) transform provided
by this package converts python types to julia types. Use [`set_jltransform!`](@ref) /
[`with_jltransform`](@ref) with a single callable to set every split at once, or with an
`AbstractDict` to set a different transform per split. [`set_format!`](@ref) /
[`reset_format!`](@ref) act on all splits.

A `DatasetDict` is an `AbstractDict{String, Dataset}`, so `keys`, `values`, `haskey`,
`get`, and iteration work as expected.

The constructors build a `DatasetDict` from in-memory Julia data — a mapping of split names
to [`Dataset`](@ref)s, given either as an `AbstractDict` or as `name => dataset` pairs. Each
split inherits its source `Dataset`'s own transform (so a dict built from `Dataset((; ...))`s
is in the `"julia"` format, the `Dataset` default); change them afterwards with
[`set_jltransform!`](@ref) or [`set_format!`](@ref). The source `Dataset`s are not mutated.

See also [`load_dataset`](@ref) and [`Dataset`](@ref).

# Examples

```jldoctest
julia> train = Dataset((; label=[1, 0, 1, 0]));

julia> test = Dataset((; label=[1, 1]));

julia> dd = DatasetDict("train" => train, "test" => test)
DatasetDict({
    train: Dataset({
        features: ['label'],
        num_rows: 4
    })
    test: Dataset({
        features: ['label'],
        num_rows: 2
    })
})

julia> collect(keys(dd))
2-element Vector{String}:
 "train"
 "test"

julia> dd["train"]
Dataset({
    features: ['label'],
    num_rows: 4
})

julia> haskey(dd, "validation")
false
```
"""
mutable struct DatasetDict <: AbstractDict{String, Dataset}
    py::Py
    jltransform::Dict{String, Any}   # per-split julia transforms, one entry per split
    # Cached, ordered split names. `keys`/`length`/iteration read this instead of calling into
    # Python, so they stay allocation-cheap AND safe to call from contexts where a Python call
    # would crash — notably the REPL's async tab-completion (`d[<TAB>` calls `keys`/`length` on
    # a task where invoking libpython segfaults). Splits are immutable for a `DatasetDict`
    # (methods that change them return a freshly-wrapped one), so the cache never goes stale.
    splits::Vector{String}

    function DatasetDict(pydatasetdict::Py, jltransform = identity)
        if !pyisinstance(pydatasetdict, datasets.DatasetDict)
            throw(ArgumentError("expected a `datasets.DatasetDict`, got $(pytype(pydatasetdict))"))
        end
        splits = _split_names(pydatasetdict)
        return new(pydatasetdict, _jltransform_dict(splits, jltransform), splits)
    end
end

# Ordered split names of a `datasets.DatasetDict`, read from Python.
_split_names(py::Py) = String[pyconvert(String, k) for k in py.keys()]

# Normalize a transform specification into a per-split `Dict` with one entry per split (given
# by the split names `ks`). A `nothing`/callable spec is broadcast to every split (`nothing`
# -> `identity`); an `AbstractDict` spec is applied per split (keys stringified), with any
# split it omits defaulting to `identity`.
function _jltransform_dict(ks::AbstractVector{<:AbstractString}, spec)
    if spec isa AbstractDict
        byname = Dict{String,Any}(String(k) => v for (k, v) in spec)
        return Dict{String,Any}(k => get(byname, k, identity) for k in ks)
    else
        t = spec === nothing ? identity : spec
        return Dict{String,Any}(k => t for k in ks)
    end
end

# Convenience for callers that only have the Python object (e.g. `set_format!`).
_jltransform_dict(py::Py, spec) = _jltransform_dict(_split_names(py), spec)

# The julia transform to record for a split supplied as a `Dataset` (its own transform) or
# a raw `Py` (none).
_jltransform_of(v::Dataset) = getfield(v, :jltransform)
_jltransform_of(::Py) = identity

DatasetDict(splits::AbstractDict{<:AbstractString, Dataset}) = _from_julia_splits(splits)

DatasetDict(splits::Pair{<:AbstractString, Dataset}...) = _from_julia_splits(splits)

# Build a `DatasetDict` from a mapping of split name to `Dataset`, mirroring `Dataset`'s
# from-Julia-data constructors. Each split's underlying Python dataset is shallow-copied
# (independent format state, shared Arrow data) so the result does not mutate the source
# `Dataset`s, and each split inherits its source's own julia transform (so a dict built
# from `Dataset((; ...))`s is in the `"julia"` format). Change the transforms afterwards
# with `set_jltransform!`/`set_format!`.
function _from_julia_splits(splits)
    py = datasets.DatasetDict()
    spec = Dict{String,Any}()
    for (k, v) in splits
        ks = String(k)
        py[ks] = pycopy.copy(_py(v))
        spec[ks] = _jltransform_of(v)
    end
    return DatasetDict(py, spec)
end

# Property names on a `DatasetDict` routed to this package's own methods instead of being
# forwarded to Python: the format methods (which understand the `"julia"` pseudo-format) and
# the julia-bridged `map`/`filter`, which both act per example over every split. `d.map(f)` is
# `map(f, d)` and `d.filter(f)` is `filter(f, d)` (see below). To hand a raw Python callback,
# use `d.py.map(...)` / `d.py.filter(...)`.
function _method_override(d::DatasetDict, s::Symbol)
    if s === :with_format
        return (args...; kws...) -> with_format(d, args...; kws...)
    elseif s === :set_format
        return (args...; kws...) -> set_format!(d, args...; kws...)
    elseif s === :reset_format
        return (args...; kws...) -> reset_format!(d, args...; kws...)
    elseif s === :map
        return (f; kws...) -> map(f, d; kws...)
    elseif s === :filter
        return (f; kws...) -> filter(f, d; kws...)
    else
        return nothing
    end
end

function Base.getproperty(d::DatasetDict, s::Symbol)
    if s in fieldnames(DatasetDict)
        return getfield(d, s)
    end
    # Route the format methods to this package's own versions (see `_method_override`) so the
    # Python-style `d.with_format(...)`/`d.set_format(...)`/`d.reset_format()` calls handle the
    # `"julia"` format; every other name forwards to the wrapped Python object.
    override = _method_override(d, s)
    override === nothing || return override
    res = getproperty(getfield(d, :py), s)
    if pycallable(res)
        return CallableWrapper(res, getfield(d, :jltransform))
    else
        return res |> py2jl
    end
end

# `length`/`keys`/`haskey` answer from the cached `splits` (no Python call): correct because
# splits are immutable, and required so the REPL's async `d[<TAB>` completion — which calls
# `keys`/`length` — does not invoke libpython off the main task and segfault.
Base.length(d::DatasetDict) = length(getfield(d, :splits))

function Base.getindex(d::DatasetDict, i::AbstractString)
    x = d.py[i]
    return Dataset(x, get(d.jltransform, i, identity))
end

Base.keys(d::DatasetDict) = getfield(d, :splits)

Base.values(d::DatasetDict) = Dataset[d[k] for k in keys(d)]

Base.pairs(d::DatasetDict) = Pair{String,Dataset}[k => d[k] for k in keys(d)]

Base.haskey(d::DatasetDict, k::AbstractString) = k in getfield(d, :splits)

Base.iterate(d::DatasetDict, state...) = iterate(pairs(d), state...)

# Single Python round-trip: `datasets.DatasetDict` subclasses `dict`, so `dict.get`
# with a `None` sentinel both tests membership and fetches the value at once. Stored
# values are always `Dataset`s, so `None` unambiguously means "absent".
function Base.get(d::DatasetDict, k::AbstractString, default)
    x = d.py.get(k, nothing)
    return pyis(x, pybuiltins.None) ? default : Dataset(x, get(d.jltransform, k, identity))
end

# The generic `AbstractDict` `merge` builds the result with `empty(d)`, which returns a plain
# `Dict` and drops the wrapper. Override it to return a `DatasetDict` backed by a fresh
# `datasets.DatasetDict`. Each split keeps its own transform (captured from the `Dataset`
# value); later dicts win for both the data and the transform, mirroring `Base.merge`.
# (`filter` is not an `AbstractDict`-style override here — see `Base.filter` below.)
_py(v::Dataset) = getfield(v, :py)
_py(v::Py) = v

function _wrap_pairs(itr)
    py = datasets.DatasetDict()
    spec = Dict{String,Any}()
    for (k, v) in itr
        ks = String(k)
        py[ks] = _py(v)
        spec[ks] = _jltransform_of(v)
    end
    return DatasetDict(py, spec)
end

function Base.merge(d::DatasetDict, others::AbstractDict...)
    return _wrap_pairs(Iterators.flatten((pairs(d), map(pairs, others)...)))
end

"""
    map(f, d::DatasetDict; kws...)

Apply `f` to every example of every split of `d` through `datasets`' `DatasetDict.map`,
bridging Julia values on both sides just like the [`Dataset`](@ref) version: each example (or
batch, with `batched=true`) is converted with [`py2jl`](@ref) before `f` sees it, and `f`'s
return value is converted back with [`jl2py`](@ref). Keyword arguments are forwarded to the
Python `map`, and each split keeps its own julia format/transform.

`d.map(f; ...)` is equivalent to this `map(f, d; ...)`; use `d.py.map(...)` for a raw Python
callback.

See also [`filter`](@ref).
"""
function Base.map(f, d::DatasetDict; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(d, :py).map(g; kws...)
    return DatasetDict(y, getfield(d, :jltransform))
end

"""
    filter(f, d::DatasetDict; kws...)

Filter every split of `d` by the Julia predicate `f`, applied per example (Python's
`DatasetDict.filter`), bridging values with [`py2jl`](@ref)/[`jl2py`](@ref) exactly as
[`map`](@ref) does. Returns a `DatasetDict` with the same splits, each keeping its own julia
format/transform. `d.filter(f; ...)` is equivalent to this `filter(f, d; ...)`.

!!! note
    This deliberately overrides the generic `AbstractDict` `filter` (which would filter split
    *entries*): `filter(f, ::DatasetDict)` filters *examples* within every split, to match
    Python and the property-style `d.filter(f)`. To select splits, index the `DatasetDict` or
    build a new one explicitly.
"""
function Base.filter(f, d::DatasetDict; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(d, :py).filter(g; kws...)
    return DatasetDict(y, getfield(d, :jltransform))
end

function Base.deepcopy_internal(d::DatasetDict, stackdict::IdDict)
    haskey(stackdict, d) && return stackdict[d]::DatasetDict
    py = pycopy.deepcopy(getfield(d, :py))
    jltransform = Base.deepcopy_internal(getfield(d, :jltransform), stackdict)
    d2 = DatasetDict(py, jltransform)
    stackdict[d] = d2
    return d2
end

# Shallow copy: the returned dict shares the underlying Arrow data with `d` but has an
# independent format, so `set_format!`/`set_jltransform!` on the copy do not affect the
# original. `copy.copy` on the `DatasetDict` alone would share the *child* `Dataset`
# objects (and hence their format state), so shallow-copy each split individually.
function Base.copy(d::DatasetDict)
    py = datasets.DatasetDict()
    for (k, v) in getfield(d, :py).items()
        py[k] = pycopy.copy(v)
    end
    return DatasetDict(py, getfield(d, :jltransform))
end

Base.show(io::IO, ds::DatasetDict) = print(io, ds.py)

# `DatasetDict` is an `AbstractDict`, so without this method the REPL would use the
# generic `AbstractDict` multi-line display. Show the Python-style repr instead, which
# mirrors `datasets.DatasetDict` and nests the `Dataset` summaries.
Base.show(io::IO, ::MIME"text/plain", ds::DatasetDict) = print(io, ds.py)

"""
    with_jltransform(d::DatasetDict, transform)
    with_jltransform(transform, d::DatasetDict)

Return a copy of `d` with the julia `transform` applied to each [`Dataset`](@ref).

`transform` may be a single callable (or `nothing`), applied to every split, or an
`AbstractDict` mapping split names to per-split transforms (splits it omits fall back to
`identity`).
"""
function with_jltransform(d::DatasetDict, transform)
    d = copy(d)
    set_jltransform!(d, transform)
    return d
end

with_jltransform(transform, d::DatasetDict) = with_jltransform(d, transform)

"""
    set_jltransform!(d::DatasetDict, transform)
    set_jltransform!(transform, d::DatasetDict)

Set the transform of `d` to `transform`. Mutating
version of [`with_jltransform`](@ref).

`transform` may be a single callable (or `nothing`), applied to every split, or an
`AbstractDict` mapping split names to per-split transforms (splits it omits fall back to
`identity`).
"""
function set_jltransform!(d::DatasetDict, transform)
    d.jltransform = _jltransform_dict(d.py, transform)
    return d
end

set_jltransform!(transform, d::DatasetDict) = set_jltransform!(d, transform)
    
"""
    with_format(d::DatasetDict, format)
    
Return a copy of `d` with the format set to `format`.
If format is `"julia"`, the returned dataset will be transformed
with [`py2jl`](@ref) and copyless conversion from python types
will be used when possible.
"""
function with_format(d::DatasetDict, format)
    d = copy(d)
    return set_format!(d, format)
end

"""
    set_format!(d::DatasetDict, format)

Set the format of `d` to `format`. Mutating
version of [`with_format`](@ref).
"""
function set_format!(d::DatasetDict, format)
    if format == "julia"
        d.py.set_format("numpy")
        d.jltransform = _jltransform_dict(d.py, py2jl)
    else
        d.py.set_format(format)
        d.jltransform = _jltransform_dict(d.py, identity)
    end
    return d
end

set_format!(d::DatasetDict) = reset_format!(d)

"""
    reset_format!(d::DatasetDict)

Reset `d` to the default `"julia"` format, i.e. `set_format!(d, "julia")`. To instead
strip all formatting and get raw Python observations, use `set_format!(d, nothing)`.
"""
reset_format!(d::DatasetDict) = set_format!(d, "julia")
