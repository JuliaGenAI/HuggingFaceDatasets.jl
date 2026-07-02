"""
    IterableDataset

A Julia wrapper around an object of the python `datasets.IterableDataset` class — the lazy,
streaming counterpart of [`Dataset`](@ref). This is what `load_dataset(...; streaming=true)`
returns for a single split.

Unlike [`Dataset`](@ref), an `IterableDataset` has **no random access and no length**: it is
consumed by iteration (`for obs in itds`, `collect(itds)`, `Iterators.take(itds, n)`), not by
indexing. Its transforms (`map`/`filter`/`shuffle(buffer_size=…)`/`take`/`skip`) are lazy and
return new `IterableDataset`s.

Provides:
- `Base.iterate` over the underlying python iterator, applying the julia transform (`py2jl`
  under the default `"julia"` format) to each yielded observation.
- All python methods of `datasets.IterableDataset`, forwarded via `getproperty` and re-wrapped
  (so `.take(n)`, `.skip(n)`, `.shuffle(buffer_size=…)`, `.map(f)` come back as
  `IterableDataset`s carrying the same julia format/transform).

`getindex`, `length`, and `firstindex`/`lastindex` are intentionally **not** supported (they
throw an explanatory `ArgumentError`): a stream has no random access. Materialize rows with
`collect`, `Iterators.take`, or the lazy `.take(n)`/`.skip(n)` methods instead.

Usually constructed via [`load_dataset`](@ref) with `streaming=true`, or from a materialized
[`Dataset`](@ref) via `ds.to_iterable_dataset()`. Defaults to the `"julia"` format so each
yielded observation is converted to native Julia types on access. A raw
`datasets.IterableDataset` can also be wrapped directly, in which case it is format-neutral
(use [`with_format`](@ref) to opt in to `"julia"`).

See also [`load_dataset`](@ref), [`Dataset`](@ref), [`IterableDatasetDict`](@ref), and
[`with_format`](@ref).

# Examples

```jldoctest
julia> ds = Dataset((; x = [1, 2, 3]));

julia> itds = ds.to_iterable_dataset();   # wrapped IterableDataset, "julia" format

julia> for obs in itds
           println(obs["x"])
       end
1
2
3

julia> for obs in itds.take(2)             # lazy `.take`, still an IterableDataset
           println(obs["x"])
       end
1
2
```
"""
mutable struct IterableDataset
    py::Py
    jltransform

    function IterableDataset(py::Py, jltransform = identity)
        if !pyisinstance(py, datasets.IterableDataset)
            throw(ArgumentError("expected a `datasets.IterableDataset`, got $(pytype(py))"))
        end
        return new(py, jltransform)
    end
end

# Property names routed to this package's own methods instead of being forwarded to Python,
# mirroring `Dataset`: the format methods understand the `"julia"` pseudo-format, and
# `map`/`filter` bridge Julia values through `py2jl`/`jl2py`. Returns a callable bound to `ds`,
# or `nothing` when `s` is not one of these (the caller then forwards to Python). To hand
# `map`/`filter` a raw Python callback, use the underlying `ds.py.map(...)`/`ds.py.filter(...)`.
function _method_override(ds::IterableDataset, s::Symbol)
    if s === :with_format
        return (args...; kws...) -> with_format(ds, args...; kws...)
    elseif s === :set_format
        return (args...; kws...) -> set_format!(ds, args...; kws...)
    elseif s === :reset_format
        return (args...; kws...) -> reset_format!(ds, args...; kws...)
    elseif s === :map
        return (f; kws...) -> map(f, ds; kws...)
    elseif s === :filter
        return (f; kws...) -> filter(f, ds; kws...)
    else
        return nothing
    end
end

function Base.getproperty(ds::IterableDataset, s::Symbol)
    if s in fieldnames(IterableDataset)
        return getfield(ds, s)
    end
    override = _method_override(ds, s)
    override === nothing || return override
    res = getproperty(getfield(ds, :py), s)
    if pycallable(res)
        return CallableWrapper(res, getfield(ds, :jltransform))
    else
        return res |> py2jl
    end
end

# A stream has no length or random access. `IteratorSize` is `SizeUnknown` so `collect` and
# `Iterators.take` work without asking for a length.
Base.IteratorSize(::Type{IterableDataset}) = Base.SizeUnknown()

# Iterate over observations, driving the underlying python iterator (a `Py` is itself iterable
# via PythonCall) and applying the julia transform (`py2jl` under the `"julia"` format) to each
# yielded row. `for obs in itds`, `collect`, and `Iterators.take` all go through this.
function Base.iterate(ds::IterableDataset)
    r = iterate(getfield(ds, :py))
    r === nothing && return nothing
    obs, state = r
    return getfield(ds, :jltransform)(obs), state
end

function Base.iterate(ds::IterableDataset, state)
    r = iterate(getfield(ds, :py), state)
    r === nothing && return nothing
    obs, state = r
    return getfield(ds, :jltransform)(obs), state
end

# No random access / no length: guide the user to the streaming interface instead of a bare
# `MethodError`.
_no_index(what) = throw(ArgumentError(
    "`IterableDataset` does not support `$what`; it is a lazy stream with no random access. " *
    "Iterate it (`for obs in itds`), or use `collect`, `Iterators.take(itds, n)`, or the lazy " *
    "`itds.take(n)` / `itds.skip(n)` methods."))

Base.length(::IterableDataset) = _no_index("length")
Base.getindex(::IterableDataset, ::Any) = _no_index("getindex")
Base.firstindex(::IterableDataset) = _no_index("firstindex")
Base.lastindex(::IterableDataset) = _no_index("lastindex")

Base.show(io::IO, ds::IterableDataset) = print(io, getfield(ds, :py))

"""
    map(f, ds::IterableDataset; kws...)

Lazily apply `f` to every example of the stream through `datasets`' `IterableDataset.map`,
bridging Julia values on both sides exactly like the [`Dataset`](@ref) version: each example
(or batch, with `batched=true`) is converted with [`py2jl`](@ref) before `f` sees it, and `f`'s
return value is converted back to Python with [`jl2py`](@ref). Nothing is materialized — the
returned `IterableDataset` applies `f` on the fly as it is iterated.

`ds.map(f; ...)` is equivalent to this `map(f, ds; ...)`; use `ds.py.map(...)` for a raw Python
callback. See also [`filter`](@ref).
"""
function Base.map(f, ds::IterableDataset; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(ds, :py).map(g; kws...)
    return IterableDataset(y, getfield(ds, :jltransform))
end

"""
    filter(f, ds::IterableDataset; kws...)

Lazily filter the stream by the Julia predicate `f` through `datasets`'
`IterableDataset.filter`, bridging values with [`py2jl`](@ref)/[`jl2py`](@ref) as [`map`](@ref)
does. `ds.filter(f; ...)` is equivalent to this `filter(f, ds; ...)`.
"""
function Base.filter(f, ds::IterableDataset; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(ds, :py).filter(g; kws...)
    return IterableDataset(y, getfield(ds, :jltransform))
end

# Shallow copy: shares the underlying python iterable but has an independent format/transform.
# Safe because `set_format!` *replaces* the `py` field (via `with_format`, which returns a new
# python object) rather than mutating it in place, so the original is untouched.
Base.copy(ds::IterableDataset) = IterableDataset(getfield(ds, :py), getfield(ds, :jltransform))

"""
    with_format(ds::IterableDataset, format)

Return a copy of `ds` with the format set to `format`. As for [`Dataset`](@ref), `"julia"` is
numpy-backed + [`py2jl`](@ref); `nothing` yields raw Python observations; any other string is
forwarded to `datasets`' own `with_format`. See also [`set_format!`](@ref).
"""
function with_format(ds::IterableDataset, format)
    ds = copy(ds)
    return set_format!(ds, format)
end

"""
    set_format!(ds::IterableDataset, format)

Set the format of `ds` to `format`. Mutating version of [`with_format`](@ref). Unlike
[`Dataset`](@ref), `datasets.IterableDataset` has no in-place `set_format`, so this replaces the
wrapped python object with `py.with_format(...)`.

`format == "julia"` installs the julia format (numpy-backed + [`py2jl`](@ref)); `nothing`
removes all formatting (raw Python observations); any other string is forwarded to `datasets`'
`with_format` (`"numpy"`, `"torch"`, ...). The single-argument form restores the julia format.
"""
function set_format!(ds::IterableDataset, format)
    if format == "julia"
        ds.py = getfield(ds, :py).with_format("numpy")
        ds.jltransform = py2jl
    else
        ds.py = getfield(ds, :py).with_format(format)
        ds.jltransform = identity
    end
    return ds
end

set_format!(ds::IterableDataset) = reset_format!(ds)

"""
    reset_format!(ds::IterableDataset)

Reset `ds` to the default `"julia"` format, i.e. `set_format!(ds, "julia")`. To instead strip
all formatting and get raw Python observations, use `set_format!(ds, nothing)`.
"""
reset_format!(ds::IterableDataset) = set_format!(ds, "julia")

"""
    with_jltransform(ds::IterableDataset, transform)
    with_jltransform(transform, ds::IterableDataset)

Return a copy of `ds` with the julia transform (applied to each yielded observation) set to
`transform`. If `transform` is `nothing` or `identity`, no transform is applied.
"""
function with_jltransform(ds::IterableDataset, transform)
    ds = copy(ds)
    return set_jltransform!(ds, transform)
end

with_jltransform(transform, ds::IterableDataset) = with_jltransform(ds, transform)

"""
    set_jltransform!(ds::IterableDataset, transform)
    set_jltransform!(transform, ds::IterableDataset)

Set the julia transform of `ds` to `transform`. Mutating version of [`with_jltransform`](@ref).
"""
function set_jltransform!(ds::IterableDataset, transform)
    ds.jltransform = transform === nothing ? identity : transform
    return ds
end

set_jltransform!(transform, ds::IterableDataset) = set_jltransform!(ds, transform)
