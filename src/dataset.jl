"""
    Dataset(pydataset; transform = identity)

A Julia wrapper around the objects of the python `datasets.Dataset` class.

The `transform` is applied after the python dataset one
that can be set with `ds.set_transform(pytransform)`.

The [`py2jl`](@ref) transform provided by this package 
converts python types to julia types.

Provides: 
- 1-based indexing.
- [`set_transform!`](@ref) julia method.
- All python class' methods from  `datasets.Dataset`.

See also [`load_dataset`](@ref) and [`DatasetDict`](@ref).
"""
mutable struct Dataset
    pyd::Py
    transform

    function Dataset(pydataset::Py; transform = identity)
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

"""
    with_format(ds::Dataset, format)

Return a copy of `ds` with the format set to `format`.
If format is `"julia"`, the returned dataset will be transformed
with [`py2jl`](@ref) and copyless conversion from python types 
will be used when possible.

See also [`set_format!`](@ref).

# Examples

```julia
julia> ds = load_dataset("mnist", split="test");

julia> ds[1]
Python dict: {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x2B5B4C1F0>, 'label': 7}

julia> ds = with_format(ds, "julia");

julia> ds[1]
Dict{String, Any} with 2 entries:
  "label" => 7
  "image" => UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
```
"""
function with_format(d::Dataset, format::AbstractString)
    if format == "julia"
        pyd = d.pyd.with_format("numpy")
        return Dataset(pyd; transform = py2jl)
    else
        return Dataset(d.pyd.with_format(format); d.transform)
    end
end

"""
    set_format!(ds::Dataset, format)

Set the format of `ds` to `format`. Mutating
version of [`with_format`](@ref).
"""
function set_format!(d::Dataset, format)
    if format == "julia"
        pyd = d.pyd.set_format("numpy")
        return Dataset(pyd; transform = py2jl)
    else
        return Dataset(d.pyd.set_format(format); d.transform)
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

