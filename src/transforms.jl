"""
    py2jl(x)

Convert Python types to Julia types applying `pyconvert` recursively.
"""
py2jl(x)

py2jl(x) = tojulia(pyconvert(x))

tojulia(x) = x
tojulia(x::PyList) = [py2jl(x) for x in x]
tojulia(x::PyDict) = Dict(py2jl(k) => py2jl(v) for (k, v) in pairs(x))
# py2julia(x::PyArray) = 
