struct CallableWrapper
    f::Py
    jltransform          # parent's julia transform, propagated onto returned datasets
end

CallableWrapper(f::Py) = CallableWrapper(f, identity)

function (cw::CallableWrapper)(args...; kwargs...)
    y = py2jl(cw.f(args...; kwargs...))
    # A forwarded method that returns a new dataset (`shuffle`, `select`, `map`, ...) drops
    # the parent's `jltransform` because `py2jl` re-wraps with the default `identity`.
    # Re-attach it so the `"julia"` format (and any custom transform) survives the call,
    # matching how `datasets` propagates Python-side formats. For a real Python format the
    # transform is `identity`, so this is a no-op and Python's own propagation still governs.
    if y isa Union{Dataset, DatasetDict, IterableDataset, IterableDatasetDict}
        return set_jltransform!(y, cw.jltransform)
    end
    return y
end

# The Python function handed to `datasets`' `map`/`filter` for the Julia callback `f`. `pyfunc`
# passes the arguments as raw `Py`s, so `py2jl` converts them; a plain Julia function would get
# PythonCall's own conversion (`PyDict`, `PyArray`, ...), on which `py2jl` is a no-op. A Python
# callable is passed as is.
_pycallback(f) =
    pyfunc((args...; kws...) -> jl2py(f(map(py2jl, args)...; map(py2jl, values(kws))...)))
_pycallback(f::Py) = f
