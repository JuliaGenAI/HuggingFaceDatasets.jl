"""
    Dataset

A Julia wrapper around an object of the python `datasets.Dataset` class.

Provides:
- 1-based indexing.
- All python class' methods from  `datasets.Dataset`.

Usually constructed via [`load_dataset`](@ref), but any `datasets.Dataset` object can
be wrapped directly.

See also [`load_dataset`](@ref), [`DatasetDict`](@ref), and [`with_format`](@ref).

# Examples

```jldoctest
julia> ds = Dataset(datasets.Dataset.from_dict(pydict(label=[5, 0, 4])))
Dataset({
    features: ['label'],
    num_rows: 3
})

julia> length(ds)
3

julia> ds[1]      # a single observation, as a Python object
Python: {'label': 5}

julia> ds[end]
Python: {'label': 4}

julia> ds = with_format(ds, "julia");   # convert observations to Julia types

julia> ds[1]
Dict{String, Int64} with 1 entry:
  "label" => 5

julia> ds[1:3]    # a range or vector returns a batch (columns -> vectors)
Dict{String, Vector{Int64}} with 1 entry:
  "label" => [5, 0, 4]
```
"""
mutable struct Dataset
    py::Py
    jltransform

    function Dataset(py::Py, jltransform = identity)
        if !pyisinstance(py, datasets.Dataset)
            throw(ArgumentError("expected a `datasets.Dataset`, got $(pytype(py))"))
        end
        return new(py, jltransform)
    end
end

## TODO make it work with arbitrary order tensors
# function Dataset(d::Dict; jltransform = identity)
#     py = datasets.Dataset.from_dict(d)
#     return Dataset(py, jltransform)
# end

function Base.getproperty(ds::Dataset, s::Symbol)
    if s in fieldnames(Dataset)
        return getfield(ds, s)
    elseif s === :with_format
        return format -> with_format(ds, format)
    else
        res = getproperty(getfield(ds, :py), s)
        if pycallable(res)
            return CallableWrapper(res, getfield(ds, :jltransform))
        else
            return res |> py2jl
        end
    end
end

Base.length(ds::Dataset) = length(ds.py)

Base.firstindex(ds::Dataset) = 1
Base.lastindex(ds::Dataset) = length(ds)

# Iterate over observations, so that `for obs in ds`, `collect(ds)`, etc. work.
function Base.iterate(ds::Dataset, state=(1, length(ds)))
    i, n = state
    i > n && return nothing
    return ds[i], (i + 1, n)
end

Base.getindex(ds::Dataset, ::Colon) = ds[1:length(ds)]

function Base.getindex(ds::Dataset, i::AbstractVector{<:Integer})
    all(≥(1), i) || throw(BoundsError(ds, i))
    x = getfield(ds, :py)[i .- 1]
    return ds.jltransform(x)
end

function Base.getindex(ds::Dataset, i::Integer)
    x = ds[[i]] # transforms and jltransforms always work on batches
    return getobs(x, 1)
end

function Base.getindex(ds::Dataset, i::AbstractString)
    x = ds.py[i]
    d = @py {i: x}
    return ds.jltransform(d)[i]
end

"""
    map(f, ds::Dataset; kws...)

Apply `f` to `ds` through `datasets`' `map`, bridging Julia values on both sides: each
example (or batch, with `batched=true`) is converted with [`py2jl`](@ref) before `f` sees
it, and `f`'s return value is converted back to Python with [`jl2py`](@ref). This lets you
write pure-Julia transforms while still getting `datasets`' batching, caching, and
multiprocessing.

Keyword arguments (`batched`, `num_proc`, `remove_columns`, ...) are forwarded to the
Python `map`. The parent's julia format/transform is preserved on the returned `Dataset`.

Use `ds.map(...)` (the forwarded Python method) if you need to hand `map` a raw Python
callback instead.

See also [`filter`](@ref).

# Examples

```julia
julia> ds = with_format(Dataset(datasets.Dataset.from_dict(pydict(label=[5, 0, 4]))), "julia");

julia> ds2 = map(x -> Dict("label" => x["label"] .+ 100), ds; batched=true);

julia> ds2[1:3]["label"]
3-element Vector{Int64}:
 105
 100
 104
```
"""
function Base.map(f, ds::Dataset; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(ds, :py).map(g; kws...)
    return Dataset(y, getfield(ds, :jltransform))
end

"""
    filter(f, ds::Dataset; kws...)

Filter `ds` through `datasets`' `filter`, bridging Julia values: each example (or batch,
with `batched=true`) is converted with [`py2jl`](@ref) before `f` sees it, and `f` returns
a `Bool` (or, when `batched=true`, a `Vector{Bool}`), converted back to Python with
[`jl2py`](@ref).

Keyword arguments are forwarded to the Python `filter`; the parent's julia format/transform
is preserved on the returned `Dataset`.

See also [`map`](@ref).
"""
function Base.filter(f, ds::Dataset; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(ds, :py).filter(g; kws...)
    return Dataset(y, getfield(ds, :jltransform))
end

function Base.deepcopy_internal(ds::Dataset, stackdict::IdDict)
    haskey(stackdict, ds) && return stackdict[ds]::Dataset
    py = pycopy.deepcopy(getfield(ds, :py))
    jltransform = Base.deepcopy_internal(getfield(ds, :jltransform), stackdict)
    ds2 = Dataset(py, jltransform)
    stackdict[ds] = ds2
    return ds2
end

# Shallow copy: the returned dataset shares the underlying Arrow data with `ds`
# but has an independent format, so `set_format!`/`set_jltransform!` on the copy
# do not affect the original. Used internally by the copy-on-write helpers.
function Base.copy(ds::Dataset)
    py = pycopy.copy(getfield(ds, :py))
    return Dataset(py, getfield(ds, :jltransform))
end

Base.show(io::IO, ds::Dataset) = print(io, ds.py)

"""
    with_format(ds::Dataset, format)

Return a copy of `ds` with the format set to `format`.
If format is `"julia"`, the returned dataset will be transformed
with [`py2jl`](@ref) and copyless conversion from python types 
will be used when possible.

See also [`set_format!`](@ref).

# Examples

```jldoctest
julia> ds = Dataset(datasets.Dataset.from_dict(pydict(label=[5, 0, 4])));

julia> ds[1]
Python: {'label': 5}

julia> ds = with_format(ds, "julia");

julia> ds[1]
Dict{String, Int64} with 1 entry:
  "label" => 5
```
"""
function with_format(ds::Dataset, format::AbstractString)
    ds = copy(ds)
    return set_format!(ds, format)
end

"""
    set_format!(ds::Dataset, format)

Set the format of `ds` to `format`. Mutating
version of [`with_format`](@ref).
"""
function set_format!(ds::Dataset, format)
    if format == "julia"
        ds.py.reset_format() # or ds.py.set_format("python")
        ds.jltransform = py2jl
    else
        ds.py.set_format(format)
        ds.jltransform = identity
    end
    return ds
end

set_format!(ds::Dataset) = reset_format!(ds)

function reset_format!(ds::Dataset)
    ds.py.set_format(nothing)
    ds.jltransform = identity
    return ds
end

"""
    with_jltransform(ds::Dataset, transform)
    with_jltransform(transform, ds::Dataset)

Return a copy of `ds` with the julia transform set to `transform`.
The `transform` applies when indexing, e.g. `ds[1]` or `ds[1:2]`.

The transform is always applied to a batch of data, even if the index is a single integer.
That is, `ds[1]` is equivalent to `ds[1:1]` from the point of view of the transform.

The julia transform is applied after the python transform (if any). 
The python transform can be set with `ds.set_transform(pytransform)`.

If `transform` is `nothing` or `identity`, the returned dataset will not be transformed.

See also [`set_jltransform!`](@ref) for the mutating version.
"""
function with_jltransform(ds::Dataset, transform)
    ds = copy(ds)
    return set_jltransform!(ds, transform)
end

# conveniency for the do syntax
with_jltransform(transform, ds::Dataset) = with_jltransform(ds, transform)

"""
    set_jltransform!(ds::Dataset, transform)
    set_jltransform!(transform, ds::Dataset)

Set the julia transform of `ds` to `transform`. Mutating
version of [`with_jltransform`](@ref).
"""
function set_jltransform!(ds::Dataset, transform)
    if transform === nothing
        ds.jltransform = identity
    else
        ds.jltransform = transform
    end
    return ds
end

set_jltransform!(transform, ds::Dataset) = set_jltransform!(ds, transform)
