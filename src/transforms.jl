
# See https://github.com/cjdoris/PythonCall.jl/issues/172.
function _pyconvert(x::Py)
    if pyisinstance(x, PIL.PngImagePlugin.PngImageFile)
        # TODO: attempt to convert to a Julia image type. 
        return numpy2jl(np.array(x))
    elseif pyisinstance(x, np.ndarray)
        return numpy2jl(x)
    else
        return pyconvert(Any, x)
    end
end

_pyconvert(x) = x

"""
    py2jl(x)

Convert Python types to Julia types applying `pyconvert` recursively.
"""
py2jl

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
    if Bool(x.dtype.type == np.str_)
        return PyArray(x, copy=false)
    else
        return DLPack.wrap(x, x -> x.__dlpack__())
    end
end
