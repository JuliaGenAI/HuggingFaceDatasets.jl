"""
    Column{T} <: AbstractVector{T}

A lazy, 1-based vector view over a single column of a [`Dataset`]. It wraps the
python `datasets.Column` object that `dataset[column_name]` returns
and converts each element from python to julia with [`py2jl`](@ref) only when it
is accessed, so the whole column is never materialized at once.

Because `Column <: AbstractVector`, it indexes, slices, iterates, broadcasts,
compares, and `collect`s like an ordinary vector; `collect(col)` materializes it
into a plain `Vector`. The element type `T` is inferred from the first element
(`Any` if the column is empty).

`collect(col)` and slicing with a range or a vector of indices (`col[a:b]`) read the
rows in one batch. Element-wise access (`col[i]`, `for x in col`, `sum(col)`, broadcasting)
costs a Python call per element, so for more than a few elements, slice or `collect`
first.

Returned by [`py2jl`](@ref) on a `datasets.Column`, and hence by string indexing
of a julia-formatted [`Dataset`], e.g. `ds["label"]`.

# Examples

```jldoctest
julia> ds = Dataset((; label=[5, 0, 4]));   # "julia" format by default

julia> col = ds["label"]
3-element HuggingFaceDatasets.Column{Int64}:
 5
 0
 4

julia> col[2]
0

julia> col[1:2]
2-element Vector{Int64}:
 5
 0

julia> collect(col)
3-element Vector{Int64}:
 5
 0
 4
```
"""
struct Column{T} <: AbstractVector{T}
    py::Py
end

function Column(py::Py)
    n = pylen(py)
    T = n == 0 ? Any : typeof(py2jl(py[0]))
    return Column{T}(py)
end

# The underlying python `datasets.Column`.
pyobj(c::Column) = getfield(c, :py)

Base.size(c::Column) = (pylen(pyobj(c)),)

function Base.getindex(c::Column, i::Integer)
    @boundscheck checkbounds(c, i)
    return py2jl(pyobj(c)[i - 1])
end

# Fetch a batch of indices in a single python round-trip instead of one per element. A range
# goes as a Python `range`, which `datasets` reads as one Arrow slice; a list of indices is
# validated and gathered one by one.
function Base.getindex(c::Column, ii::AbstractVector{<:Integer})
    @boundscheck checkbounds(c, ii)
    key = ii isa AbstractRange ? ii .- 1 : pylist(Int[i - 1 for i in ii])
    return py2jl(pyobj(c)[key])
end

# Logical indexing (`Bool <: Integer`, so this must be handled separately).
function Base.getindex(c::Column, mask::AbstractVector{Bool})
    @boundscheck length(mask) == length(c) || throw(BoundsError(c, mask))
    return c[findall(mask)]
end

# Nested columns (struct features): `col["field"]` returns another lazy `Column`.
Base.getindex(c::Column, name::AbstractString) = Column(pyobj(c)[name])

# Read the whole column as one batch: iterating the python column would format and convert
# it row by row. A fixed-size array column comes back stacked along its last axis, so split
# it into one array per row, as for the rows of a `RaggedVector`.
function Base.collect(c::Column{T}) where {T}
    isempty(c) && return T[]
    x = c[1:length(c)]
    x isa RaggedVector && return map(collect, x)
    x isa AbstractVector && return x
    x isa AbstractArray && return [copy(selectdim(x, ndims(x), j)) for j in axes(x, ndims(x))]
    return py2jl(pylist(pyobj(c)))   # a batch `py2jl` can't convert, e.g. from a torch format
end
