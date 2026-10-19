"""
    BatchIterator

The iterator returned by `ds.iter(; batch_size, drop_last_batch = false, release = true)` for a
[`Dataset`](@ref) or an [`IterableDataset`](@ref), which yields `ds`'s rows in batches of
`batch_size` (the last one shorter, unless `drop_last_batch = true`) by driving `datasets`'
`iter`. It can be iterated more than once, and for a `Dataset` it has a `length`.

Each batch is what indexing `ds` with a range returns, so in the default `"julia"` format, a
`Dict{String}` of Julia columns, e.g. a `Vector{String}` for a text column. The Python batch is
released as soon as it has been converted.

With a Python format (`nothing`, `"numpy"`, ...), each batch is a [`PyBatch`](@ref), which is
valid until the iterator advances: then it is released, along with the columns taken from it,
so that a loop over a stream holds one batch in memory at a time. `copy` or [`py2jl`](@ref) one
to keep it. Pass `release = false` to leave the batches to Julia's GC instead, e.g. to collect
them, or to iterate ahead of their use (`Iterators.Stateful`, a prefetching `Channel`, ...).

With a custom [`with_jltransform`](@ref) transform, each Python batch is passed to it, and is
released by Julia's GC.

# Examples

```jldoctest
julia> ds = Dataset((; text = ["a", "bb", "ccc"], n = Int32[1, 2, 3]));

julia> batches = ds.iter(batch_size = 2);

julia> length(batches)
2

julia> for batch in batches
           println(batch["text"], " ", batch["n"])
       end
["a", "bb"] Int32[1, 2]
["ccc"] Int32[3]
```
"""
struct BatchIterator{N<:Union{Int, Nothing}}
    py::Py
    batch_size::Int
    drop_last_batch::Bool
    release::Bool
    jltransform
    length::N
end

function _batch_iterator(ds::Union{Dataset, IterableDataset}, batch_size::Integer;
                         drop_last_batch::Bool = false, release::Bool = true)
    batch_size ≥ 1 || throw(ArgumentError("`batch_size` must be positive, got $batch_size"))
    n = ds isa Dataset ? (drop_last_batch ? fld : cld)(length(ds), batch_size) : nothing
    return BatchIterator(getfield(ds, :py), Int(batch_size), drop_last_batch, release,
                         getfield(ds, :jltransform), n)
end

_batch_iterator(ds::Union{Dataset, IterableDataset}; batch_size::Integer, kws...) =
    _batch_iterator(ds, batch_size; kws...)

Base.IteratorSize(::Type{BatchIterator{Int}}) = Base.HasLength()
Base.IteratorSize(::Type{BatchIterator{Nothing}}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:BatchIterator}) = Base.EltypeUnknown()
Base.length(it::BatchIterator{Int}) = it.length

function Base.iterate(it::BatchIterator)
    pyit = pyiter(it.py.iter(; batch_size = it.batch_size, drop_last_batch = it.drop_last_batch))
    return iterate(it, (pyit, nothing))
end

function Base.iterate(it::BatchIterator, (pyit, previous))
    previous === nothing || _release!(previous)
    PythonCall.pyisnull(pyit) && return nothing
    batch = PythonCall.unsafe_pynext(pyit)
    if PythonCall.pyisnull(batch)
        PythonCall.pydel!(pyit)
        return nothing
    end
    if it.jltransform === py2jl
        x = py2jl(batch)
        # `py2jl` builds a new `Dict` from a `dict`, but it may wrap another batch object (e.g. a
        # Python transform's `UserDict`) without copying it.
        pyisinstance(batch, pybuiltins.dict) && PythonCall.pydel!(batch)
        return x, (pyit, nothing)
    elseif it.jltransform === identity
        x = PyBatch(batch)
        return x, (pyit, it.release ? x : nothing)
    else
        return it.jltransform(batch), (pyit, nothing)
    end
end

Base.show(io::IO, it::BatchIterator) =
    print(io, "BatchIterator(batch_size = ", it.batch_size, ", drop_last_batch = ",
          it.drop_last_batch, ")")

"""
    PyBatch

A batch of a dataset in a Python format, yielded by its [`BatchIterator`](@ref): the Python
batch (a `dict` of columns for most formats), with `batch[column]` returning a column as a
`Py`, and `Py(batch)` the batch itself, e.g. to pass it to a Python function.

Iterating a `PyBatch` gives `column => batch[column]` pairs, as for a `Dict`.

When the iterator advances, the batch and the columns taken from it are released, i.e. they
become Python `None`, freeing their memory right away rather than when Julia's GC runs (unless
the iterator was created with `release = false`). `copy(batch)` returns a `PyBatch` that stays
valid, and [`py2jl`](@ref) converts it to Julia. Arrays that [`numpy2jl`](@ref) made from a
column stay valid too.
"""
mutable struct PyBatch
    const py::Py
    const columns::Vector{Py}
    released::Bool
end

PyBatch(py::Py) = PyBatch(py, Py[], false)

function _checkvalid(b::PyBatch)
    b.released && throw(ArgumentError(
        "this batch was released when its batch iterator advanced; `copy` the batch, or " *
        "convert it with `py2jl`, to keep it, or create the iterator with `release = false`"))
    return b.py
end

function Base.getindex(b::PyBatch, k)
    column = _checkvalid(b)[k]
    push!(b.columns, column)
    return column
end

Base.keys(b::PyBatch) = pyconvert(Vector{String}, pylist(_checkvalid(b).keys()))
Base.haskey(b::PyBatch, k) = pycontains(_checkvalid(b), k)
Base.length(b::PyBatch) = length(keys(b))

function Base.iterate(b::PyBatch, (ks, i) = (keys(b), 1))
    i > length(ks) && return nothing
    return ks[i] => b[ks[i]], (ks, i + 1)
end

Base.copy(b::PyBatch) = PyBatch(PythonCall.pynew(_checkvalid(b)))

# A new reference, since `py2jl` may wrap the batch object rather than copy it.
py2jl(b::PyBatch) = py2jl(PythonCall.pynew(_checkvalid(b)))

PythonCall.ispy(::PyBatch) = true
PythonCall.Py(b::PyBatch) = _checkvalid(b)

function Base.show(io::IO, mime::MIME"text/plain", b::PyBatch)
    if b.released
        print(io, "PyBatch (released)")
    else
        println(io, "PyBatch:")
        show(io, mime, b.py)
    end
end

Base.show(io::IO, b::PyBatch) =
    b.released ? print(io, "PyBatch(released)") : print(io, "PyBatch(", b.py, ")")

# Points `x` at `None`, releasing what it referenced. Unlike after `pydel!`, every Julia
# reference to `x` stays a valid Python object.
function _release!(x::Py)
    PythonCall.pydel!(x)
    PythonCall.pycopy!(x, pybuiltins.None)
    return nothing
end

function _release!(b::PyBatch)
    b.released && return nothing
    b.released = true
    foreach(_release!, b.columns)
    empty!(b.columns)
    _release!(b.py)
    return nothing
end
