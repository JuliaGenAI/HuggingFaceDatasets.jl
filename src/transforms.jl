"""
    py2jl(x)

Convert Python types to Julia types applying `pyconvert` recursively.
"""
py2jl

py2jl(x) = tojulia(pyconvert(x))

tojulia(x) = x

function tojulia(x::Py)
    if pyisinstance(x, PIL.PngImagePlugin.PngImageFile)
        return np.array(x) |> pyconvert |> Array
    else
        return x
    end
end

tojulia(x::PyList) = [py2jl(x) for x in x]
tojulia(x::PyDict) = Dict(py2jl(k) => py2jl(v) for (k, v) in pairs(x))