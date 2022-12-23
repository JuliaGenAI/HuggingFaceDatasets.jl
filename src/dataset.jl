"""
    Dataset(pydataset; transform = py2jl)

A Julia wrapper around the objects of the python `datasets.Dataset` class.

The `transform` is applied after datasets' one. 
The [`py2jl`](@ref) default converts python types to julia types.

Provides: 
- 1-based indexing.
- [`set_transform!`](@ref) julia method.
- All python class' methods from  `datasets.Dataset`.

See also [`load_dataset`](@ref) and [`DatasetDict`](@ref).
"""
mutable struct Dataset
    pyd::Py
    transform

    function Dataset(pydataset::Py; transform = py2jl)
        @assert pyisinstance(pydataset, datasets.Dataset)
        return new(pydataset, transform)
    end
end

function Base.getproperty(d::Dataset, s::Symbol)
    if s in fieldnames(Dataset)
        return getfield(d, s)
    else
        res = getproperty(getfield(d, :pyd), s)
        if pyisinstance(res, datasets.Dataset)
            return Dataset(res; d.transform)
        else
            return res |> py2jl
        end
    end
end

Base.length(d::Dataset) = length(d.pyd)

Base.getindex(d::Dataset, ::Colon) = d[1:length(d)]

function Base.getindex(d::Dataset, i::AbstractVector{<:Integer})
    @assert all(>(0), i)
    x = d.pyd[i .- 1]
    return d.transform(x)
end

function Base.getindex(d::Dataset, i::Integer)
    x = d[[i]] # transforms always work on batches
    return getobs(x, 1) 
end

function Base.getindex(d::Dataset, i::AbstractString)
    x = d.pyd[i]
    return d.transform(pydict(Dict(i =>x)))[i]
end

function set_transform!(d::Dataset, transform)
    if transform === nothing
        d.transform = identity
    else
        d.transform = transform
    end
end

