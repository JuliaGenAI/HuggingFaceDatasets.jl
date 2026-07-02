"""
    IterableDatasetDict

A dictionary of [`IterableDataset`](@ref)s — the streaming counterpart of [`DatasetDict`](@ref),
wrapping a `datasets.IterableDatasetDict`. This is what `load_dataset(...; streaming=true)`
returns when no `split` is selected.

Like [`DatasetDict`](@ref) it is an `AbstractDict{String, IterableDataset}` (so `keys`, `values`,
`haskey`, `get`, and iteration work) and stores a julia transform **per split**: indexing a split
(`dd["train"]`) hands back an [`IterableDataset`](@ref) carrying that split's transform.
[`set_format!`](@ref) / [`reset_format!`](@ref) act on all splits.

See also [`load_dataset`](@ref) and [`IterableDataset`](@ref).
"""
mutable struct IterableDatasetDict <: AbstractDict{String, IterableDataset}
    py::Py
    jltransform::Dict{String, Any}   # per-split julia transforms, one entry per split
    # Cached, ordered split names — read by `keys`/`length`/iteration instead of calling into
    # Python, so they stay allocation-cheap and safe from the REPL's async tab-completion (see
    # the analogous note on `DatasetDict`). Splits are immutable, so the cache never goes stale.
    splits::Vector{String}

    function IterableDatasetDict(py::Py, jltransform = identity)
        if !pyisinstance(py, datasets.IterableDatasetDict)
            throw(ArgumentError("expected a `datasets.IterableDatasetDict`, got $(pytype(py))"))
        end
        splits = _split_names(py)
        return new(py, _jltransform_dict(splits, jltransform), splits)
    end
end

function _method_override(d::IterableDatasetDict, s::Symbol)
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

function Base.getproperty(d::IterableDatasetDict, s::Symbol)
    if s in fieldnames(IterableDatasetDict)
        return getfield(d, s)
    end
    override = _method_override(d, s)
    override === nothing || return override
    res = getproperty(getfield(d, :py), s)
    if pycallable(res)
        return CallableWrapper(res, getfield(d, :jltransform))
    else
        return res |> py2jl
    end
end

Base.length(d::IterableDatasetDict) = length(getfield(d, :splits))

function Base.getindex(d::IterableDatasetDict, i::AbstractString)
    x = getfield(d, :py)[i]
    return IterableDataset(x, get(getfield(d, :jltransform), i, identity))
end

Base.keys(d::IterableDatasetDict) = getfield(d, :splits)

Base.values(d::IterableDatasetDict) = IterableDataset[d[k] for k in keys(d)]

Base.pairs(d::IterableDatasetDict) = Pair{String,IterableDataset}[k => d[k] for k in keys(d)]

Base.haskey(d::IterableDatasetDict, k::AbstractString) = k in getfield(d, :splits)

Base.iterate(d::IterableDatasetDict, state...) = iterate(pairs(d), state...)

function Base.get(d::IterableDatasetDict, k::AbstractString, default)
    haskey(d, k) || return default
    return d[k]
end

Base.show(io::IO, d::IterableDatasetDict) = print(io, getfield(d, :py))
Base.show(io::IO, ::MIME"text/plain", d::IterableDatasetDict) = print(io, getfield(d, :py))

"""
    map(f, d::IterableDatasetDict; kws...)

Lazily apply `f` to every example of every split, bridging Julia values on both sides like the
[`IterableDataset`](@ref) version. `d.map(f; ...)` is equivalent to this `map(f, d; ...)`.
"""
function Base.map(f, d::IterableDatasetDict; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(d, :py).map(g; kws...)
    return IterableDatasetDict(y, getfield(d, :jltransform))
end

"""
    filter(f, d::IterableDatasetDict; kws...)

Lazily filter every split by the Julia predicate `f`, applied per example, bridging values with
[`py2jl`](@ref)/[`jl2py`](@ref). Like [`DatasetDict`](@ref)'s `filter`, this filters *examples*
within every split (matching Python), not split entries.
"""
function Base.filter(f, d::IterableDatasetDict; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(d, :py).filter(g; kws...)
    return IterableDatasetDict(y, getfield(d, :jltransform))
end

"""
    with_format(d::IterableDatasetDict, format)

Return a copy of `d` with the format set to `format` on every split (see
[`with_format`](@ref)).
"""
function with_format(d::IterableDatasetDict, format)
    d = copy(d)
    return set_format!(d, format)
end

"""
    set_format!(d::IterableDatasetDict, format)

Set the format of every split of `d` to `format`. Mutating version of [`with_format`](@ref).
"""
function set_format!(d::IterableDatasetDict, format)
    if format == "julia"
        d.py = getfield(d, :py).with_format("numpy")
        d.jltransform = _jltransform_dict(getfield(d, :py), py2jl)
    else
        d.py = getfield(d, :py).with_format(format)
        d.jltransform = _jltransform_dict(getfield(d, :py), identity)
    end
    return d
end

set_format!(d::IterableDatasetDict) = reset_format!(d)

"""
    reset_format!(d::IterableDatasetDict)

Reset `d` to the default `"julia"` format, i.e. `set_format!(d, "julia")`.
"""
reset_format!(d::IterableDatasetDict) = set_format!(d, "julia")

# Shallow copy sharing the python object; safe because `set_format!` replaces `py` rather than
# mutating it.
Base.copy(d::IterableDatasetDict) = IterableDatasetDict(getfield(d, :py), getfield(d, :jltransform))

"""
    with_jltransform(d::IterableDatasetDict, transform)
    with_jltransform(transform, d::IterableDatasetDict)

Return a copy of `d` with the julia `transform` applied to each [`IterableDataset`](@ref).
`transform` may be a single callable (or `nothing`), applied to every split, or an
`AbstractDict` mapping split names to per-split transforms.
"""
function with_jltransform(d::IterableDatasetDict, transform)
    d = copy(d)
    return set_jltransform!(d, transform)
end

with_jltransform(transform, d::IterableDatasetDict) = with_jltransform(d, transform)

"""
    set_jltransform!(d::IterableDatasetDict, transform)
    set_jltransform!(transform, d::IterableDatasetDict)

Set the transform of `d` to `transform`. Mutating version of [`with_jltransform`](@ref).
"""
function set_jltransform!(d::IterableDatasetDict, transform)
    d.jltransform = _jltransform_dict(getfield(d, :py), transform)
    return d
end

set_jltransform!(transform, d::IterableDatasetDict) = set_jltransform!(d, transform)
