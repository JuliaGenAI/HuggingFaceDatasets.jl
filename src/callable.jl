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
    if y isa Dataset || y isa DatasetDict
        return set_jltransform!(y, cw.jltransform)
    end
    return y
end
