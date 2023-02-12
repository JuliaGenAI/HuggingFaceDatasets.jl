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
    elseif s === :with_format
        return format -> with_format(d, format)
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

function Base.deepcopy(d::DatasetDict)
    pyd = copy.deepcopy(d.pyd)
    return DatasetDict(pyd, d.jltransform)
end

Base.show(io::IO, ds::DatasetDict) = print(io, ds.pyd)

""""
    with_jltransform(d::DatasetDict, transform)
    with_jltransform(transform, d::DatasetDict)

Return a copy of `d` with the julia `transform` applied to each [`Dataset`](@ref).
"""
function with_jltransform(d::DatasetDict, transform)
    d = deepcopy(d)
    set_jltransform!(d, transform)
    return d
end

with_jltransform(transform, d::DatasetDict) = with_jltransform(d, transform)

"""
    set_jltransform!(d::DatasetDict, transform)
    set_jltransform!(transform, d::DatasetDict)

Set the transform of `d` to `transform`. Mutating 
version of [`with_jltransform`](@ref).
"""
function set_jltransform!(d::DatasetDict, transform)
    if transform === nothing
        d.jltransform = identity
    else
        d.jltransform = transform
    end
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
    d = deepcopy(d)
    return set_format!(d, format)
end

"""
    set_format!(d::DatasetDict, format)

Set the format of `d` to `format`. Mutating
version of [`with_format`](@ref).
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
