
"""
    py2jl(x)

Convert Python types to Julia types. It will recursively traverse built-in python
containers such as lists, tuples, dicts, and sets, and convert all nested objects.
On the leaves, it will call either `pyconvert(Any, x)` or [`numpy2jl`](@ref).
"""
py2jl(x) = pyconvert(Any, x)

function py2jl(x::Py)
    # handle datasets
    if pyisinstance(x, datasets.Dataset)
        return Dataset(x)
    elseif pyisinstance(x, datasets.DatasetDict)
        return DatasetDict(x)
    # handle list, tuple, dict, and set
    elseif pyisinstance(x, pytype(pylist()))
        return [py2jl(x) for x in x]
    elseif pyisinstance(x, pytype(pytuple()))
        return tuple(py2jl(x) for x in x)
    elseif pyisinstance(x, pytype(pydict()))
        return Dict(py2jl(k) => py2jl(v) for (k, v) in x.items())
    elseif pyisinstance(x, pytype(pyset()))
        return Set(py2jl(x) for x in x)
    # handle numpy arrays   
    elseif pyisinstance(x, np.ndarray)
        return numpy2jl(x)
    # handle PIL images
    elseif pyisinstance(x, PIL.PngImagePlugin.PngImageFile) || pyisinstance(x, PIL.JpegImagePlugin.JpegImageFile)
        a = numpy2jl(np.array(x))
        if ndims(a) == 3 && size(a, 1) == 3
            return colorview(RGB{N0f8}, a)
        elseif ndims(a) == 2
            return reinterpret(Gray{N0f8}, a)
        else
            error("Unknown image format")
        end
    # handle other types
    else
        return pyconvert(Any, x)
    end
end


"""
    numpy2jl(x)

Convert a numpy array to a Julia array using DLPack.jl.
The conversion is copyless, and mutations to the Julia array are reflected in the numpy array.
For row major python arrays, the returned Julia array has permuted dimensions.

This function is called by [`py2jl`](@ref).
See also [`jl2numpy`](@ref).
"""
function numpy2jl(x::Py)
    return DLPack.from_dlpack(x)
end

"""
    jl2numpy(x)

Convert a Julia array to a numpy array using DLPack.jl.
The conversion is copyless, and mutations to the numpy array are reflected in the Julia array.
The returned numpy array has permuted dimensions with respect to the input Julia array.

See also [`numpy2jl`](@ref).
"""
function jl2numpy(x::AbstractArray)
    return DLPack.share(x, np.from_dlpack)
end
