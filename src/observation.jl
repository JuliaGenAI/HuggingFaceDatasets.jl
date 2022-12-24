
function MLUtils.getobs(xpy::Py, i::Integer)
    if pyisinstance(xpy, pytype(pydict()))
        return pydict(Dict(k => getobs(v, i) for (k,v) in xpy.items()))
    elseif pyisinstance(xpy, pytype(pylist()))
        # TODO do this only for lists containing numbers
        return xpy[i-1] 
    elseif pyisinstance(xpy, np.ndarray)
        return xpy[i-1]
    else
        return error("Py type $(pytype(xpy)) non supported yet")
    end
end
