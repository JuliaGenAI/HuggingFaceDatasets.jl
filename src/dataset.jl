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
    pyds::Py
    jltransform

    function Dataset(pyds::Py, jltransform = identity)
        if !pyisinstance(pyds, datasets.Dataset)
            throw(ArgumentError("expected a `datasets.Dataset`, got $(pytype(pyds))"))
        end
        return new(pyds, jltransform)
    end
end

## TODO make it work with arbitrary order tensors
# function Dataset(d::Dict; jltransform = identity)
#     pyds = datasets.Dataset.from_dict(d)
#     return Dataset(pyds, jltransform)
# end

function Base.getproperty(ds::Dataset, s::Symbol)
    if s in fieldnames(Dataset)
        return getfield(ds, s)
    elseif s === :with_format
        return format -> with_format(ds, format)
    else
        res = getproperty(getfield(ds, :pyds), s)
        if pycallable(res)
            return CallableWrapper(res)
        else
            return res |> py2jl
        end
    end
end

Base.length(ds::Dataset) = length(ds.pyds)

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
    x = getfield(ds, :pyds)[i .- 1]
    return ds.jltransform(x)
end

function Base.getindex(ds::Dataset, i::Integer)
    x = ds[[i]] # transforms and jltransforms always work on batches
    return getobs(x, 1)
end

function Base.getindex(ds::Dataset, i::AbstractString)
    x = ds.pyds[i]
    d = @py {i: x}
    return ds.jltransform(d)[i]
end

function Base.deepcopy_internal(ds::Dataset, stackdict::IdDict)
    haskey(stackdict, ds) && return stackdict[ds]::Dataset
    pyds = pycopy.deepcopy(getfield(ds, :pyds))
    jltransform = Base.deepcopy_internal(getfield(ds, :jltransform), stackdict)
    ds2 = Dataset(pyds, jltransform)
    stackdict[ds] = ds2
    return ds2
end

# Shallow copy: the returned dataset shares the underlying Arrow data with `ds`
# but has an independent format, so `set_format!`/`set_jltransform!` on the copy
# do not affect the original. Used internally by the copy-on-write helpers.
function Base.copy(ds::Dataset)
    pyds = pycopy.copy(getfield(ds, :pyds))
    return Dataset(pyds, getfield(ds, :jltransform))
end

Base.show(io::IO, ds::Dataset) = print(io, ds.pyds)

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
        ds.pyds.reset_format() # or d.pyd.set_format("python")
        ds.jltransform = py2jl
    else
        ds.pyds.set_format(format)
        ds.jltransform = identity
    end
    return ds
end

set_format!(ds::Dataset) = reset_format!(ds)

function reset_format!(ds::Dataset)
    ds.pyds.set_format(nothing)
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
