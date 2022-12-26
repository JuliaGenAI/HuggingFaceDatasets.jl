
function MLUtils.getobs(py::Py, i::Integer)
    if pyisinstance(py, pytype(pydict()))
        return pydict(Dict(k => getobs(v, i) for (k, v) in py.items()))
    elseif pyisinstance(py, pytype(pylist()))
        # TODO do this only for lists containing numbers
        return py[i-1] 
    elseif pyisinstance(xpy, np.ndarray)
        return py[i-1]
    else
        return error("Py type $(pytype(py)) non supported yet")
    end
end
