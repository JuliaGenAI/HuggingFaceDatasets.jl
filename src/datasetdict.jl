"""
    DatasetDict(pydatasetdict::Py; transform = identity)

A `DatasetDict` is a dictionary of `Dataset`s. 
It is a wrapper around a `datasets.DatasetDict` object.

The `transform` is applied to each [`Dataset`](@ref). 
The [`py2jl`](@ref) transform provided by this package
converts python types to julia types.

See also [`load_dataset`](@ref) and [`Dataset`](@ref).
"""
mutable struct DatasetDict
    pyd::Py
    transform

    function DatasetDict(pydatasetdict::Py; transform = identity)
        @assert pyisinstance(pydatasetdict, datasets.DatasetDict)
        return new(pydatasetdict, transform)
    end
end

function Base.getproperty(d::DatasetDict, s::Symbol)
    if s in fieldnames(DatasetDict)
        return getfield(d, s)
    else
        res = getproperty(getfield(d, :pyd), s)
        if pyisinstance(res, datasets.Dataset)
            return Dataset(res; d.transform)
        elseif pyisinstance(res, datasets.DatasetDict)
            return DatasetDict(res; d.transform)
        else
            return res |> py2jl
        end
    end
end

Base.length(d::DatasetDict) = length(d.pyd)

function Base.getindex(d::DatasetDict, i::AbstractString)
    x = d.pyd[i]
    return Dataset(x; d.transform)
end

function set_transform!(d::DatasetDict, transform)
    if transform === nothing
        d.transform = identity
    else
        d.transform = transform
    end
end

function with_format(d::DatasetDict, format)
    if format == "julia"
        pyd = d.pyd.with_format("numpy")
        return DatasetDict(pyd; transform = py2jl)
    else 
        pyd = d.pyd.with_format(format)
        return DatasetDict(pyd; d.transform)
    end
end
    
"""
    set_format!(d::DatasetDict, format)

Set the format of `d` to `format`.
If format is `"julia"`, the returned dataset will be transformed
with [`py2jl`](@ref) and copyless conversion from python types
will be used when possible.
"""
function set_format!(d::DatasetDict, format)
    if format == "julia"
        d.pyd.set_format("numpy")
        d.transform = py2jl
    else
        d.pyd.set_format(format)
    end
    return d
end
