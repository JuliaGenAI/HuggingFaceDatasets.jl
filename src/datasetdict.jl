"""
    DatasetDict(py::Py, jltransform = identity)

A `DatasetDict` is a dictionary of `Dataset`s.
It is a wrapper around a `datasets.DatasetDict` object.

The `jltransform` is applied to each [`Dataset`](@ref).
The [`py2jl`](@ref) transform provided by this package
converts python types to julia types.

A `DatasetDict` is an `AbstractDict{String, Dataset}`, so `keys`, `values`, `haskey`,
`get`, and iteration work as expected.

See also [`load_dataset`](@ref) and [`Dataset`](@ref).

# Examples

```jldoctest
julia> train = datasets.Dataset.from_dict(pydict(label=[1, 0, 1, 0]));

julia> test = datasets.Dataset.from_dict(pydict(label=[1, 1]));

julia> dd = DatasetDict(datasets.DatasetDict(pydict(; train, test)))
DatasetDict({
    train: Dataset({
        features: ['label'],
        num_rows: 4
    })
    test: Dataset({
        features: ['label'],
        num_rows: 2
    })
})

julia> collect(keys(dd))
2-element Vector{String}:
 "train"
 "test"

julia> dd["train"]
Dataset({
    features: ['label'],
    num_rows: 4
})

julia> haskey(dd, "validation")
false
```
"""
mutable struct DatasetDict <: AbstractDict{String, Dataset}
    py::Py
    jltransform

    function DatasetDict(pydatasetdict::Py, jltransform = identity)
        if !pyisinstance(pydatasetdict, datasets.DatasetDict)
            throw(ArgumentError("expected a `datasets.DatasetDict`, got $(pytype(pydatasetdict))"))
        end
        return new(pydatasetdict, jltransform)
    end
end

function Base.getproperty(d::DatasetDict, s::Symbol)
    if s in fieldnames(DatasetDict)
        return getfield(d, s)
    elseif s === :with_format
        return format -> with_format(d, format)
    else
        res = getproperty(getfield(d, :py), s)
        if pycallable(res)
            return CallableWrapper(res)
        else
            return res |> py2jl
        end
    end
end

Base.length(d::DatasetDict) = length(d.py)

function Base.getindex(d::DatasetDict, i::AbstractString)
    x = d.py[i]
    return Dataset(x, d.jltransform)
end

Base.keys(d::DatasetDict) = String[pyconvert(String, k) for k in d.py.keys()]

Base.values(d::DatasetDict) = Dataset[d[k] for k in keys(d)]

Base.pairs(d::DatasetDict) = Pair{String,Dataset}[k => d[k] for k in keys(d)]

Base.haskey(d::DatasetDict, k::AbstractString) = pyconvert(Bool, pyin(k, d.py))

Base.iterate(d::DatasetDict, state...) = iterate(pairs(d), state...)

# Single Python round-trip: `datasets.DatasetDict` subclasses `dict`, so `dict.get`
# with a `None` sentinel both tests membership and fetches the value at once. Stored
# values are always `Dataset`s, so `None` unambiguously means "absent".
function Base.get(d::DatasetDict, k::AbstractString, default)
    x = d.pyd.get(k, nothing)
    return pyis(x, pybuiltins.None) ? default : Dataset(x, d.jltransform)
end

# The generic `AbstractDict` `merge`/`filter` build the result with `empty(d)`, which
# returns a plain `Dict` and drops the wrapper. Override them to return a `DatasetDict`
# backed by a fresh `datasets.DatasetDict`, preserving `d`'s `jltransform`.
_py(v::Dataset) = getfield(v, :py)
_py(v::Py) = v

function _wrap_pairs(itr, jltransform)
    py = datasets.DatasetDict()
    for (k, v) in itr
        py[String(k)] = _py(v)
    end
    return DatasetDict(py, jltransform)
end

Base.filter(f, d::DatasetDict) = _wrap_pairs(Iterators.filter(f, pairs(d)), d.jltransform)

function Base.merge(d::DatasetDict, others::AbstractDict...)
    return _wrap_pairs(Iterators.flatten((pairs(d), map(pairs, others)...)), d.jltransform)
end

function Base.deepcopy_internal(d::DatasetDict, stackdict::IdDict)
    haskey(stackdict, d) && return stackdict[d]::DatasetDict
    py = pycopy.deepcopy(getfield(d, :py))
    jltransform = Base.deepcopy_internal(getfield(d, :jltransform), stackdict)
    d2 = DatasetDict(py, jltransform)
    stackdict[d] = d2
    return d2
end

# Shallow copy: the returned dict shares the underlying Arrow data with `d`
# but has an independent format, so `set_format!`/`set_jltransform!` on the copy
# do not affect the original. Used internally by the copy-on-write helpers.
function Base.copy(d::DatasetDict)
    py = pycopy.copy(getfield(d, :py))
    return DatasetDict(py, getfield(d, :jltransform))
end

Base.show(io::IO, ds::DatasetDict) = print(io, ds.py)

# `DatasetDict` is an `AbstractDict`, so without this method the REPL would use the
# generic `AbstractDict` multi-line display. Show the Python-style repr instead, which
# mirrors `datasets.DatasetDict` and nests the `Dataset` summaries.
Base.show(io::IO, ::MIME"text/plain", ds::DatasetDict) = print(io, ds.py)

"""
    with_jltransform(d::DatasetDict, transform)
    with_jltransform(transform, d::DatasetDict)

Return a copy of `d` with the julia `transform` applied to each [`Dataset`](@ref).
"""
function with_jltransform(d::DatasetDict, transform)
    d = copy(d)
    set_jltransform!(d, transform)
    return d
end

with_jltransform(transform, d::DatasetDict) = with_jltransform(d, transform)

"""
    set_jltransform!(d::DatasetDict, transform)
    set_jltransform!(transform, d::DatasetDict)

Set the transform of `d` to `transform`. Mutating 
version of [`with_jltransform`](@ref).
"""
function set_jltransform!(d::DatasetDict, transform)
    if transform === nothing
        d.jltransform = identity
    else
        d.jltransform = transform
    end
    return d
end

set_jltransform!(transform, d::DatasetDict) = set_jltransform!(d, transform)
    
"""
    with_format(d::DatasetDict, format)
    
Return a copy of `d` with the format set to `format`.
If format is `"julia"`, the returned dataset will be transformed
with [`py2jl`](@ref) and copyless conversion from python types
will be used when possible.
"""
function with_format(d::DatasetDict, format)
    d = copy(d)
    return set_format!(d, format)
end

"""
    set_format!(d::DatasetDict, format)

Set the format of `d` to `format`. Mutating
version of [`with_format`](@ref).
"""
function set_format!(d::DatasetDict, format)
    if format == "julia"
        d.py.reset_format()
        d.jltransform = py2jl
    else
        d.py.set_format(format)
        d.jltransform = identity
    end
    return d
end

set_format!(d::DatasetDict) = reset_format!(d)

"""
    reset_format!(d::DatasetDict)

Reset the format of `d`, removing any python format and julia transform.
"""
function reset_format!(d::DatasetDict)
    d.py.set_format(nothing)
    d.jltransform = identity
    return d
end
