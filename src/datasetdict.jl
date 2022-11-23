"""
    DatasetDict
"""
mutable struct DatasetDict
    pyd::Py
    transform

    function DatasetDict(pydatasetdict::Py; transform = py2jl)
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

