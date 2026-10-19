# Content fingerprints of the Julia functions given to `Dataset` `map`/`filter`.
#
# `datasets` finds a cached transform result by a fingerprint of the transform, computed by
# pickling the function. A Julia function pickles by name, so a redefined function, or a
# changed global, would reuse a stale result. Instead the function is handed to `datasets`
# wrapped in a `JuliaFunction` (see `python/huggingfacedatasets_jl.py`) that pickles as a hash
# of what the function does, computed here like `dill` does for a Python function: the lowered
# code of the function and of the code it uses from your own modules, the values it captures
# and the globals it reads, and the identities of the packages it calls into.

struct _Unhashable <: Exception
    msg::String
end

mutable struct _Hasher
    ctx::SHA.SHA1_CTX
    io::IOBuffer
    ids::IdDict{Any,Int}            # visited objects, written again as back-references
    exts::IdDict{Any,Nothing}       # functions of other modules whose methods from your code are hashed
    modids::IdDict{Module,Any}      # `_module_id` of each module seen
    mods::Dict{Int,Module}          # statements of the code being hashed that evaluate to a module
end
_Hasher() = _Hasher(SHA.SHA1_CTX(), IOBuffer(), IdDict{Any,Int}(), IdDict{Any,Nothing}(),
                    IdDict{Module,Any}(), Dict{Int,Module}())

_flush!(h::_Hasher) = SHA.update!(h.ctx, take!(h.io))
_tag!(h::_Hasher, c::Char) = write(h.io, UInt8(c))
_int!(h::_Hasher, n::Integer) = write(h.io, Int64(n))
_str!(h::_Hasher, s::AbstractString) = (_int!(h, ncodeunits(s)); write(h.io, s))
_digest!(h::_Hasher) = (_flush!(h); bytes2hex(SHA.digest!(h.ctx)))

# Write a back-reference if `x` was already hashed; otherwise record it and return `false`.
function _seen!(h::_Hasher, @nospecialize(x))
    i = get(h.ids, x, 0)
    i == 0 || (_tag!(h, 'R'); _int!(h, i); return true)
    h.ids[x] = length(h.ids) + 1
    return false
end

_isdefinedglobal(m::Module, s::Symbol) =
    @static isdefined(Base, :isdefinedglobal) ? isdefinedglobal(m, s) : isdefined(m, s)

# How code in module `m` is identified: `nothing` for your own code (`Main`, modules of your
# scripts), hashed by its lowered code; otherwise a string naming the package, Julia's version
# for `Base` and the standard libraries, and the contents of `src/` for a developed package.
const _PACKAGE_KINDS = IdDict{Module,Any}()
const _PACKAGE_KINDS_LOCK = ReentrantLock()

function _module_id(h::_Hasher, m::Module)
    root = Base.moduleroot(m)
    return get!(h.modids, root) do
        kind = lock(() -> get!(() -> _package_kind(root), _PACKAGE_KINDS, root), _PACKAGE_KINDS_LOCK)
        kind isa Tuple ? _source_hash(kind...) : kind
    end
end

function _package_kind(root::Module)
    (root === Core || root === Base) && return "julia $VERSION $root"
    id = Base.PkgId(root)
    id.uuid === nothing && return nothing
    dir = pkgdir(root)
    dir === nothing && return "julia $VERSION $(id.uuid)"   # bundled with Julia, e.g. `Compiler`
    dir = normpath(dir)
    if any(d -> startswith(dir, joinpath(normpath(d), "packages", "")), DEPOT_PATH)
        return "$(id.uuid) $(pkgversion(root)) $(basename(dir))"   # the slug follows the content
    elseif startswith(dir, joinpath(normpath(Sys.STDLIB), ""))
        return "julia $VERSION $(id.uuid)"
    end
    return (id, dir)   # developed or tracking a path: its files may change during the session
end

function _source_hash(id::Base.PkgId, dir::String)
    ctx = SHA.SHA1_CTX()
    for sub in filter(isdir, joinpath.(dir, ("src", "ext"))), (root, _, files) in walkdir(sub)
        occursin("__pycache__", root) && continue
        for file in sort(files)
            path = joinpath(root, file)
            SHA.update!(ctx, codeunits(relpath(path, dir)))
            SHA.update!(ctx, read(path))
        end
    end
    isfile(joinpath(dir, "Project.toml")) && SHA.update!(ctx, read(joinpath(dir, "Project.toml")))
    return "$(id.uuid) $(bytes2hex(SHA.digest!(ctx)))"
end

function _hash!(h::_Hasher, @nospecialize(x))
    if x isa Dataset
        py = getfield(x, :py)
        _tag!(h, 'S'); _hash_py!(h, pytuple((py._fingerprint, py.format)))
        _hash!(h, getfield(x, :jltransform))
    elseif x isa DatasetDict
        _tag!(h, 'K'); _int!(h, length(x))
        for k in sort!(collect(keys(x)))
            _str!(h, k); _hash!(h, x[k])
        end
    elseif x isa Union{IterableDataset, IterableDatasetDict}
        throw(_Unhashable("an `$(nameof(typeof(x)))`"))
    elseif PythonCall.ispy(x)
        _hash_py!(h, Py(x))
    elseif x isa DataType
        _hash_datatype!(h, x)
    elseif x isa Module
        _tag!(h, 'M'); _str!(h, something(_module_id(h, x), "")); _str!(h, join(fullname(x), '.'))
    elseif x isa Symbol
        _tag!(h, 'Y'); _str!(h, String(x))
    elseif x isa String
        _tag!(h, 'T'); _str!(h, x)
    elseif x isa Method
        _hash_method!(h, x)
    elseif x isa Core.CodeInfo
        _hash_code!(h, x)
    elseif x isa GlobalRef
        _hash_binding!(h, x.mod, x.name)
    elseif x isa Expr
        b = _getproperty_binding(h, x)
        if b === nothing
            _tag!(h, 'E'); _str!(h, String(x.head)); _int!(h, length(x.args))
            for a in x.args
                _hash!(h, a)
            end
        else
            _hash_binding!(h, b...)
        end
    elseif x isa QuoteNode
        _tag!(h, 'Q'); _hash!(h, x.value)
    elseif x isa LineNumberNode
        # source locations don't change what the code does
    elseif x isa Core.SimpleVector
        _tag!(h, 'V'); _int!(h, length(x))
        for a in x
            _hash!(h, a)
        end
    elseif x isa Regex
        _tag!(h, 'r'); _str!(h, x.pattern); _int!(h, x.compile_options); _int!(h, x.match_options)
    elseif x isa BigInt
        _tag!(h, 'I'); _str!(h, string(x))
    elseif x isa BigFloat
        _tag!(h, 'F'); _int!(h, precision(x)); _str!(h, string(x))
    elseif x isa Ptr
        x == C_NULL || throw(_Unhashable("a pointer"))
        _tag!(h, '0')
    elseif x isa Union{Task, Channel, IO, Base.AbstractLock, Base.GenericCondition,
                       Core.TypeName, Core.MethodTable, Core.MethodInstance, Core.CodeInstance}
        throw(_Unhashable("a `$(nameof(typeof(x)))`"))
    elseif x isa AbstractDict || x isa AbstractSet
        _hash_collection!(h, x)
    elseif x isa Array || (@static isdefined(Core, :GenericMemory) ? x isa Core.GenericMemory : false)
        _hash_array!(h, x)
    else
        _hash_value!(h, x)
    end
    position(h.io) > 1 << 20 && _flush!(h)
    return
end

# Any other value: its type, then its fields.
function _hash_value!(h::_Hasher, @nospecialize(x))
    T = typeof(x)
    if isprimitivetype(T)
        _hash_datatype!(h, T)
        _tag!(h, 'b'); unsafe_write(h.io, Ref(x), sizeof(T))
        return
    end
    ismutable(x) && _seen!(h, x) && return
    _hash_datatype!(h, T)
    x isa Function && Base.issingletontype(T) && _hash_extensions!(h, x)
    for i in 1:nfields(x)
        isdefined(x, i) ? _hash!(h, getfield(x, i)) : _tag!(h, 'u')
    end
end

function _hash_py!(h::_Hasher, x::Py)
    s = try
        pyconvert(String, datasets.fingerprint.Hasher.hash(x))
    catch e
        e isa PyException || rethrow()
        throw(_Unhashable("a Python `$(pyconvert(String, pytype(x).__name__))` that `datasets` can't hash"))
    end
    _tag!(h, 'p'); _str!(h, s)
end

# Iteration order of `Dict`s and `Set`s depends on hashes that can differ between sessions.
function _hash_collection!(h::_Hasher, x)
    ismutable(x) && _seen!(h, x) && return
    _hash_datatype!(h, typeof(x))
    ks = collect(x isa AbstractDict ? keys(x) : x)
    if x isa Union{Dict, IdDict, Set}
        ks = try sort(ks) catch; ks end
    end
    _int!(h, length(ks))
    for k in ks
        _hash!(h, k)
        x isa AbstractDict && _hash!(h, x[k])
    end
end

function _hash_array!(h::_Hasher, A)
    _seen!(h, A) && return
    _hash_datatype!(h, typeof(A))
    for n in size(A)
        _int!(h, n)
    end
    if isprimitivetype(eltype(A))
        _flush!(h)
        GC.@preserve A SHA.update!(h.ctx, unsafe_wrap(Array, Ptr{UInt8}(pointer(A)), sizeof(A)))
    else
        for i in eachindex(A)
            isassigned(A, i) ? _hash!(h, A[i]) : _tag!(h, 'u')
            position(h.io) > 1 << 20 && _flush!(h)
        end
    end
end

# A type of a package is identified by its name; one of your own code by what it is (its
# generated name, e.g. a closure's `var"#5#6"`, can differ between sessions), including the
# methods that construct or call it.
function _hash_datatype!(h::_Hasher, T::DataType)
    _seen!(h, T) && return
    id = _module_id(h, T.name.module)
    name = String(T.name.name)
    if id === nothing
        _tag!(h, 'U')
        startswith(name, '#') || _str!(h, name)
    else
        _tag!(h, 'D'); _str!(h, id); _str!(h, name)
    end
    _int!(h, length(T.parameters))
    for p in T.parameters
        _hash!(h, p)
    end
    id === nothing || return
    _hash!(h, supertype(T))
    if !isabstracttype(T)
        _hash!(h, Core.svec(fieldnames(T)...)); _hash!(h, Core.svec(fieldtypes(T)...))
    end
    Base.has_free_typevars(T) && return
    _hash_methods!(h, Tuple{Type{T}, Vararg{Any}})
    _hash_methods!(h, Tuple{T, Vararg{Any}})
end

# The methods from your own code that a call with signature `sig` could dispatch to.
function _hash_methods!(h::_Hasher, @nospecialize(sig))
    ms = Base._methods_by_ftype(sig, -1, Base.get_world_counter())
    ms isa AbstractVector || throw(_Unhashable("the methods of `$sig`"))
    for match in ms
        m = match.method
        _module_id(h, m.module) === nothing && _hash_method!(h, m)
    end
end

function _hash_method!(h::_Hasher, m::Method)
    _seen!(h, m) && return
    isdefined(m, :generator) && throw(_Unhashable("a generated function, `$(m.name)`"))
    _tag!(h, 'm'); _hash!(h, m.sig); _int!(h, m.nargs); _int!(h, m.isva)
    _hash_code!(h, Base.uncompressed_ir(m))
end

function _hash_code!(h::_Hasher, ci::Core.CodeInfo)
    outer, h.mods = h.mods, Dict{Int,Module}()
    _tag!(h, 'C'); _int!(h, length(ci.code))
    for (i, stmt) in enumerate(ci.code)
        b = stmt isa GlobalRef ? (stmt.mod, stmt.name) :
            stmt isa Expr ? _getproperty_binding(h, stmt) : nothing
        if b !== nothing && _isdefinedglobal(b...) && (v = getglobal(b...)) isa Module
            h.mods[i] = v
        end
        _hash!(h, stmt)
    end
    h.mods = outer
end

# `M.x` for a module `M`, lowered (in Julia ≥ 1.12) to a call `getproperty(M, :x)` whose
# result is the binding `M.x`.
function _getproperty_binding(h::_Hasher, x::Expr)
    (x.head === :call && length(x.args) == 3) || return nothing
    f, m, s = x.args
    (f isa GlobalRef && _isdefinedglobal(f.mod, f.name) &&
        getglobal(f.mod, f.name) === Base.getproperty) || return nothing
    (s isa QuoteNode && s.value isa Symbol) || return nothing
    mod = m isa Module ? m :
          m isa Core.SSAValue ? get(h.mods, m.id, nothing) :
          m isa GlobalRef && _isdefinedglobal(m.mod, m.name) ? getglobal(m.mod, m.name) : nothing
    return mod isa Module ? (mod, s.value) : nothing
end

# A global used by the code: one of your own is hashed by its current value, one of a package
# by its name (state inside packages is not tracked).
function _hash_binding!(h::_Hasher, mod::Module, name::Symbol)
    id = _module_id(h, mod)
    defined = _isdefinedglobal(mod, name)
    if id === nothing
        _tag!(h, 'g')
        defined ? _hash!(h, getglobal(mod, name)) : (_tag!(h, '?'); _str!(h, String(name)))
    else
        _tag!(h, 'G'); _str!(h, id); _str!(h, join(fullname(mod), '.')); _str!(h, String(name))
    end
    defined && _hash_extensions!(h, getglobal(mod, name))
end

# The methods that your own code adds to a function, or constructor, of another module.
function _hash_extensions!(h::_Hasher, @nospecialize(v))
    v === Core.kwcall && return   # every keyword call: the callee is hashed instead
    T = v isa Type ? Base.unwrap_unionall(v) : typeof(v)
    (T isa DataType && (v isa Type || Base.issingletontype(T))) || return
    _module_id(h, T.name.module) === nothing && return
    haskey(h.exts, v) && return
    h.exts[v] = nothing
    _hash_methods!(h, v isa Type ? Tuple{Type{v}, Vararg{Any}} : Tuple{T, Vararg{Any}})
end

const _UNHASHED_KWARGS = (:load_from_cache_file, :cache_file_name, :desc, :num_proc,
                          :suffix_template)

# The content hash of a transform `f` called with keyword arguments `kws`, or the
# `_Unhashable` error saying why there is none.
function _content_hash(f, kws)
    h = _Hasher()
    try
        _str!(h, "HuggingFaceDatasets.jl transform")
        _str!(h, pyconvert(String, datasets.__version__))
        _hash!(h, (@__MODULE__, PythonCall))   # how values are converted on the way
        _hash!(h, f)
        for (k, v) in sort!(collect(pairs(kws)); by = first)
            k in _UNHASHED_KWARGS && continue
            _str!(h, String(k)); _hash!(h, v)
        end
        return _digest!(h)
    catch e
        e isa InterruptException && rethrow()
        return e isa _Unhashable ? e : _Unhashable(sprint(showerror, e))
    end
end

_random_fingerprint() = datasets.fingerprint.generate_random_fingerprint()

# The callable and `new_fingerprint` to hand `datasets`' `map`/`filter` for `f` on `ds`.
function _transform(f, ds::Dataset, new_fingerprint, kws)
    g = _pycallback(f)
    (f isa Py || new_fingerprint !== nothing) && return g, new_fingerprint
    # `datasets` caches only a file-backed dataset's transforms
    py = getfield(ds, :py)
    (pytruth(datasets.fingerprint.is_caching_enabled()) && pytruth(py.cache_files)) ||
        return g, _random_fingerprint()
    fp = _content_hash(f, kws)
    fp isa String && return pyhfds.JuliaFunction(g, fp), nothing
    @info "`map`/`filter` can't fingerprint the Julia function ($(fp.msg)), so its result is \
           recomputed on every call. Pass `new_fingerprint` to cache it." maxlog=1
    return g, _random_fingerprint()
end
