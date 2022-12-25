struct CallableWrapper
    f::Py
end

function (cw::CallableWrapper)(args...; kwargs...)
    y = cw.f(args..., kwargs...)
    return py2jl(y)
end
