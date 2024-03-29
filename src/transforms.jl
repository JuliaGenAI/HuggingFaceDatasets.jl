
# See https://github.com/cjdoris/PythonCall.jl/issues/172.
function _pyconvert(x::Py)
    if pyisinstance(x, datasets.Dataset)
        return Dataset(x)
    elseif pyisinstance(x, datasets.DatasetDict)
        return DatasetDict(x)
    elseif pyisinstance(x, PIL.PngImagePlugin.PngImageFile) || pyisinstance(x, PIL.JpegImagePlugin.JpegImageFile)
        a = numpy2jl(np.array(x))
        if ndims(a) == 3 && size(a, 1) == 3
            return colorview(RGB{N0f8}, a)
        elseif ndims(a) == 2
            return reinterpret(Gray{N0f8}, a)
        else
            error("Unknown image format")
        end
    elseif pyisinstance(x, np.ndarray)
        return numpy2jl(x)
    else
        return pyconvert(Any, x)
    end
end

# Do nothing on a non-Py object.
_pyconvert(x) = x

"""
    py2jl(x)

Convert Python types to Julia types applying `pyconvert` recursively.
"""
py2jl

# py2jl recurses through pycanonicalize and converts through _pyconvert
py2jl(x) = pycanonicalize(_pyconvert(x))

pycanonicalize(x) = x

pycanonicalize(x::PyList) = [py2jl(x) for x in x]
pycanonicalize(x::PyDict) = Dict(py2jl(k) => py2jl(v) for (k, v) in pairs(x))

"""
    numpy2jl(x)

Convert a numpy array to a Julia array using DLPack.
The conversion is copyless, and mutations to the Julia array are reflected in the numpy array.
"""
function numpy2jl(x::Py)
    # pyconvert(Any, x)
    # PyArray(x, copy=false)
    if Bool(x.dtype.type == np.str_)
        return PyArray(x, copy=false)
    else
        return DLPack.wrap(x, x -> x.__dlpack__())
    end
end

## TODO this doesn't work yet.
## https://github.com/pabloferz/DLPack.jl/issues/32
# function jl2numpy(x::AbstractArray)
#     return DLPack.share(x, np.from_dlpack)
# end
