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
    jltransform

    function DatasetDict(pydatasetdict::Py, jltransform = identity)
        @assert pyisinstance(pydatasetdict, datasets.DatasetDict)
        return new(pydatasetdict, jltransform)
    end
end

function Base.getproperty(d::DatasetDict, s::Symbol)
    if s in fieldnames(DatasetDict)
        return getfield(d, s)
    else
        res = getproperty(getfield(d, :pyd), s)
        if pycallable(res)
            return CallableWrapper(res)
        else
            return res |> py2jl
        end
    end
end

Base.length(d::DatasetDict) = length(d.pyd)

function Base.getindex(d::DatasetDict, i::AbstractString)
    x = d.pyd[i]
    return Dataset(x, d.jltransform)
end

function with_jltransform(d::DatasetDict, transform)
    d = deepcopy(d)
    set_jltransform!(d, transform)
    return d
end

function set_jltransform!(d::DatasetDict, transform)
    if transform === nothing
        d.transform = identity
    else
        d.transform = transform
    end
    return d
end

function with_format(d::DatasetDict, format)
    d = deepcopy(d)
    return set_format!(d, format)
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
        d.jltransform = py2jl
    else
        d.pyd.set_format(format)
        d.jltransform = identity
    end
    return d
end
