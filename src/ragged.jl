"""
    RaggedVector{T} <: AbstractVector

A vector of variable-length vectors of `T`, stored as one flat vector `values` and Arrow-style
`offsets`: `length(r) + 1` 0-based positions, so that element `i` is the view
`values[offsets[i]+1:offsets[i+1]]`.

The `"julia"` format returns a batch of a list column of numbers this way when its rows have
different lengths, e.g. the token ids of a tokenized text column, so that reading it costs two
zero-copy conversions rather than one per row. Rows of equal length are stacked into a matrix
instead, as for any array column.

`r.values` and `r.offsets` are the underlying buffers, e.g. to write out all the tokens of a
batch at once.

# Examples

```jldoctest
julia> ds = Dataset((; ids = [UInt16[1, 2, 3], UInt16[4], UInt16[5, 6]]));

julia> r = ds[1:3]["ids"];

julia> r[3]
2-element view(::Vector{UInt16}, 5:6) with eltype UInt16:
 0x0005
 0x0006

julia> r.values, r.offsets
(UInt16[0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006], Int32[0, 3, 4, 6])
```
"""
struct RaggedVector{T, V<:DenseVector{T}, O<:AbstractVector{<:Integer}} <:
       AbstractVector{SubArray{T, 1, V, Tuple{UnitRange{Int}}, true}}
    values::V
    offsets::O

    function RaggedVector(values::V, offsets::O) where {T, V<:DenseVector{T}, O<:AbstractVector{<:Integer}}
        Base.require_one_based_indexing(values, offsets)
        !isempty(offsets) && first(offsets) == 0 && last(offsets) == length(values) ||
            throw(ArgumentError("`offsets` must run from 0 to `length(values)`"))
        return new{T, V, O}(values, offsets)
    end
end

Base.size(r::RaggedVector) = (length(r.offsets) - 1,)

Base.IndexStyle(::Type{<:RaggedVector}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(r::RaggedVector, i::Int)
    @boundscheck checkbounds(r, i)
    return view(r.values, Int(r.offsets[i]) + 1:Int(r.offsets[i + 1]))
end
