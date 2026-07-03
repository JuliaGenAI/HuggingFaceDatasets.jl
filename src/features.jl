# A Julia view over a dataset's schema: `datasets.Features` and its leaf feature types
# (`ClassLabel`, `Value`). These are the schema objects returned by `ds.features`; they are
# NOT row data, so they are handled at the access site (`Dataset`'s `getproperty`, and the
# `features`/`class_names`/`int2str`/`str2int` helpers) and never touched by the generic
# `py2jl` batch hot path.
#
# Each type is `Py`-backed and forwards attribute/method access to the wrapped Python object
# (mirroring `Dataset`/`Column`), so the full Python surface (`.names`, `.num_classes`,
# `.int2str`, `.dtype`, ...) stays reachable under the same names, with results re-wrapped by
# `py2jl`. The matching `jl2py` overloads (in `transforms.jl`) unwrap them back to `.py`, so a
# view built or fetched in Julia can be handed straight back to Python (e.g. a `features=`
# schema argument).

"""
    ClassLabel(; names, num_classes)

A Julia view over a `datasets.ClassLabel` feature: the integer-encoded label type whose
`names` map class ids to human-readable strings.

Construct one from Julia (`ClassLabel(names=["neg", "pos"])`) — forwarding to
`datasets.ClassLabel` — or obtain one from a dataset's schema via `ds.features["label"]` (see
[`features`](@ref)). Attribute and method access forwards to Python, so `cl.names`,
`cl.num_classes`, `cl.int2str(i)`, and `cl.str2int(s)` all work, with results converted by
[`py2jl`](@ref).

Label integers are **0-based class ids** (data, not 1-based Julia indices): `int2str`/`str2int`
pass them through to Python unchanged. See also [`class_names`](@ref), [`int2str`](@ref),
[`str2int`](@ref), and [`Features`](@ref).

# Examples

```jldoctest
julia> ds = Dataset((; label=["cat", "dog", "dog"], x=[1, 2, 3]));

julia> ds = ds.class_encode_column("label");   # string column -> ClassLabel

julia> cl = ds.features["label"]
ClassLabel(names=['cat', 'dog'])

julia> cl.names
2-element Vector{String}:
 "cat"
 "dog"

julia> cl.int2str(1)     # 0-based class id -> name
"dog"

julia> cl.str2int("cat")
0
```
"""
struct ClassLabel
    py::Py
end

function ClassLabel(; names = nothing, num_classes = nothing)
    # Convert `names` to a Python list: a Julia `Vector` passed straight through as a kwarg
    # stays an unconverted Julia object, which later breaks `datasets`' JSON schema encoding.
    names === nothing || (names = jl2py(names))
    return ClassLabel(datasets.ClassLabel(; names, num_classes))
end

"""
    Value(dtype::AbstractString)

A Julia view over a `datasets.Value` feature: a scalar column type carrying an Arrow
`dtype` (e.g. `"int64"`, `"float32"`, `"string"`). Construct one with `Value("int64")`
(forwarding to `datasets.Value`) or obtain it from a schema via [`features`](@ref). Attribute
access forwards to Python, so `v.dtype` returns the dtype string (e.g. `ds.features["x"]`).

See also [`Features`](@ref) and [`ClassLabel`](@ref).
"""
struct Value
    py::Py
end

Value(dtype::AbstractString) = Value(datasets.Value(dtype))

# Forward attribute/method access on a leaf view to the wrapped Python object, re-wrapping the
# result with `py2jl` (callables become a `CallableWrapper`, like `Dataset`'s `getproperty`).
for T in (:ClassLabel, :Value)
    @eval function Base.getproperty(x::$T, s::Symbol)
        s === :py && return getfield(x, :py)
        res = getproperty(getfield(x, :py), s)
        return pycallable(res) ? CallableWrapper(res) : py2jl(res)
    end
    @eval Base.show(io::IO, x::$T) = print(io, getfield(x, :py))
end

"""
    Features(schema::AbstractDict)

A Julia view over a `datasets.Features` schema: an ordered mapping from column name to its
feature type. `Features <: AbstractDict{String, Any}`, so it indexes, iterates, and supports
`keys`/`values`/`haskey`/`get` like a dict; indexing a column returns the wrapped leaf
([`ClassLabel`](@ref), [`Value`](@ref)) when recognized, or the raw `Py` otherwise (nested
features, `Image`, `Audio`, `Sequence`, ...).

Obtained from a dataset via `ds.features` (or the [`features`](@ref) function), or built from
Julia (`Features(Dict("label" => ClassLabel(names=["neg", "pos"])))`) and passed back to Python
as a `features=` schema argument.

The column names are cached at construction, so `keys`/`length`/iteration never call Python
(safe from the REPL's async `feat[<TAB>` completion, mirroring [`DatasetDict`](@ref)).

# Examples

```jldoctest
julia> ds = Dataset((; label=[0, 1, 1], x=[1.0, 2.0, 3.0]));

julia> f = ds.features;

julia> collect(keys(f))
2-element Vector{String}:
 "label"
 "x"

julia> f["x"]
Value('float64')
```
"""
struct Features <: AbstractDict{String, Any}
    py::Py
    names::Vector{String}   # cached, ordered column names (Python-free keys/length/iteration)
end

function Features(py::Py)
    pyisinstance(py, datasets.Features) ||
        throw(ArgumentError("expected a `datasets.Features`, got $(pytype(py))"))
    return Features(py, String[pyconvert(String, k) for k in py.keys()])
end

Features(schema::AbstractDict) = Features(datasets.Features(jl2py(schema)))

# Wrap a schema leaf: recognized feature types get a Julia view; everything else (nested
# `Features`, `Sequence`, `Image`, `Audio`, ...) is left as a raw `Py` on purpose.
function _wrap_feature(x::Py)
    if pyisinstance(x, datasets.ClassLabel)
        return ClassLabel(x)
    elseif pyisinstance(x, datasets.Value)
        return Value(x)
    elseif pyisinstance(x, datasets.Features)
        return Features(x)
    else
        return x
    end
end

Base.getindex(f::Features, k::AbstractString) = _wrap_feature(getfield(f, :py)[k])
Base.getindex(f::Features, k::Symbol) = f[string(k)]

# `keys`/`length`/`haskey`/iteration answer from the cached `names` (no Python call).
Base.keys(f::Features) = getfield(f, :names)
Base.length(f::Features) = length(getfield(f, :names))
Base.haskey(f::Features, k) = string(k) in getfield(f, :names)

function Base.iterate(f::Features, state = 1)
    names = getfield(f, :names)
    state > length(names) && return nothing
    k = names[state]
    return (k => f[k], state + 1)
end

# Show the Python schema repr (like `Dataset`/`DatasetDict`) rather than the generic
# `AbstractDict` multi-line display.
Base.show(io::IO, f::Features) = print(io, getfield(f, :py))
Base.show(io::IO, ::MIME"text/plain", f::Features) = print(io, getfield(f, :py))

"""
    features(ds::Dataset)

Return the schema of `ds` as a [`Features`](@ref) view (also reachable as `ds.features`).
Indexing a column yields its feature type, with [`ClassLabel`](@ref)/[`Value`](@ref) leaves
wrapped for Julian access.

See also [`class_names`](@ref), [`int2str`](@ref), and [`str2int`](@ref).

# Examples

```jldoctest
julia> ds = Dataset((; label=[0, 1, 1], x=[1, 2, 3]));

julia> features(ds)["label"]
Value('int64')
```
"""
features(ds::Dataset) = Features(getfield(ds, :py).features)

# The `ClassLabel` for column `col`, erroring clearly when the column is not one.
function _classlabel(ds::Dataset, col)
    f = features(ds)[string(col)]
    f isa ClassLabel && return f
    throw(ArgumentError("column \"$col\" is not a ClassLabel feature; got $(_featkind(f))"))
end

_featkind(f::Value) = "a Value feature"
_featkind(f) = "$(typeof(f))"

"""
    class_names(ds::Dataset, col)

The ordered class names of column `col`'s [`ClassLabel`](@ref) feature, as a `Vector{String}`;
`names[i]` is the name of class id `i - 1` (ids are 0-based). Errors if `col` is not a
`ClassLabel`. Equivalent to the Pythonic `ds.features[col].names`.

See also [`int2str`](@ref), [`str2int`](@ref), and [`features`](@ref).

# Examples

```jldoctest
julia> ds = Dataset((; label=["cat", "dog", "dog"]));

julia> ds = ds.class_encode_column("label");

julia> class_names(ds, "label")
2-element Vector{String}:
 "cat"
 "dog"
```
"""
class_names(ds::Dataset, col) = pyconvert(Vector{String}, _classlabel(ds, col).py.names)

"""
    int2str(ds::Dataset, col, i)

Decode 0-based class id(s) `i` (an integer or a vector of integers) to class name(s) via
column `col`'s [`ClassLabel`](@ref), so **no index offset is applied**. Errors if `col` is not
a `ClassLabel`. Equivalent to the Pythonic `ds.features[col].int2str(i)`.

See also [`str2int`](@ref), [`class_names`](@ref), and [`features`](@ref).

# Examples

```jldoctest
julia> ds = Dataset((; label=["cat", "dog", "dog"]));

julia> ds = ds.class_encode_column("label");

julia> int2str(ds, "label", 1)
"dog"

julia> int2str(ds, "label", [0, 1, 1])
3-element Vector{String}:
 "cat"
 "dog"
 "dog"
```
"""
int2str(ds::Dataset, col, i) = py2jl(_classlabel(ds, col).py.int2str(jl2py(i)))

"""
    str2int(ds::Dataset, col, s)

Encode class name(s) `s` (a string or a vector of strings) to their 0-based class id(s) via
column `col`'s [`ClassLabel`](@ref). Errors if `col` is not a `ClassLabel`. Equivalent to the
Pythonic `ds.features[col].str2int(s)`.

See also [`int2str`](@ref), [`class_names`](@ref), and [`features`](@ref).

# Examples

```jldoctest
julia> ds = Dataset((; label=["cat", "dog", "dog"]));

julia> ds = ds.class_encode_column("label");

julia> str2int(ds, "label", "dog")
1
```
"""
str2int(ds::Dataset, col, s) = py2jl(_classlabel(ds, col).py.str2int(jl2py(s)))

# `jl2py` for the schema views: unwrap back to the underlying Python object so a
# `Features`/`ClassLabel`/`Value` built or fetched in Julia can be handed back to Python
# (e.g. a `features=` schema argument). Defined here, after the types; `jl2py`'s other methods
# live in `transforms.jl`.
jl2py(x::Features)   = getfield(x, :py)
jl2py(x::ClassLabel) = getfield(x, :py)
jl2py(x::Value)      = getfield(x, :py)
