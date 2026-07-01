
"""
    py2jl(x)

Convert Python types to Julia types. It will recursively traverse built-in python
containers such as lists, tuples, dicts, and sets, and convert all nested objects.
On the leaves, it will call either `pyconvert(Any, x)` or [`numpy2jl`](@ref).

A `datasets.Column` (the lazy column view returned by `dataset[column_name]`) is
wrapped in a lazy [`Column`](@ref), whose elements are converted on access rather
than all at once.

# Examples

```jldoctest
julia> py2jl(pylist([1, 2, 3]))
3-element Vector{Int64}:
 1
 2
 3

julia> py2jl(pytuple((1, pylist([2, 3]))))
(1, [2, 3])
```
"""
py2jl(x) = pyconvert(Any, x)

function py2jl(x::Py)
    # handle datasets
    if pyisinstance(x, datasets.Dataset)
        return Dataset(x)
    elseif pyisinstance(x, datasets.DatasetDict)
        return DatasetDict(x)
    # handle datasets.Column (the lazy column view that `dataset[column_name]`
    # returns in datasets >= 4) by wrapping it in a lazy `Column` rather than
    # materializing it here
    elseif pyisinstance(x, datasets.Column)
        return Column(x)
    # handle list, tuple, dict, and set
    elseif pyisinstance(x, pytype(pylist()))
        return [py2jl(x) for x in x]
    elseif pyisinstance(x, pytype(pytuple()))
        return Tuple(py2jl(el) for el in x)
    elseif pyisinstance(x, pytype(pydict()))
        return Dict(py2jl(k) => py2jl(v) for (k, v) in x.items())
    elseif pyisinstance(x, pytype(pyset()))
        return Set(py2jl(x) for x in x)
    # handle numpy arrays   
    elseif pyisinstance(x, np.ndarray)
        return numpy2jl(x)
    # handle PIL images (any subclass: PNG/JPEG/BMP/GIF/TIFF/... and transform outputs)
    elseif pyisinstance(x, PIL.Image.Image)
        a = numpy2jl(np.array(x))
        if ndims(a) == 3 && size(a, 1) == 3
            return colorview(RGB{N0f8}, a)
        elseif ndims(a) == 2
            return reinterpret(Gray{N0f8}, a)
        else
            # other modes (e.g. RGBA, CMYK, palette): return the raw (permuted) array
            return a
        end
    # handle other types
    else
        return pyconvert(Any, x)
    end
end


"""
    numpy2jl(x)

Convert a numpy array to a Julia array using DLPack.jl.
The conversion is copyless, and mutations to the Julia array are reflected in the numpy array.
For row major python arrays, the returned Julia array has permuted dimensions.

This function is called by [`py2jl`](@ref).
See also [`jl2numpy`](@ref).

# Examples

```jldoctest
julia> y = jl2numpy([1 2 3; 4 5 6]);   # a 3×2 numpy array

julia> numpy2jl(y)                      # back to a 2×3 Julia array
2×3 Matrix{Int64}:
 1  2  3
 4  5  6
```
"""
function numpy2jl(x::Py)
    return DLPack.from_dlpack(x)
end

"""
    jl2numpy(x)

Convert a Julia array to a numpy array, sharing memory via the buffer protocol.
The conversion is copyless, and mutations to the numpy array are reflected in the
Julia array (and vice versa). The returned numpy array has permuted dimensions with
respect to the input Julia array, since numpy is row-major and Julia is column-major.

See also [`numpy2jl`](@ref).

# Examples

```jldoctest
julia> x = [1 2 3; 4 5 6];   # a 2×3 Julia array

julia> y = jl2numpy(x);      # numpy is row-major, so the axes are reversed

julia> y.shape
Python: (3, 2)

julia> numpy2jl(y) == x
true
```
"""
function jl2numpy(x::AbstractArray)
    # `np.asarray(Py(x))` exposes `x`'s memory to numpy through the buffer protocol
    # without copying, yielding a writable view with the same shape but column-major
    # (F-contiguous) strides. `.T` reverses the axes so the result matches the
    # dimension permutation of `numpy2jl`, i.e. `numpy2jl(jl2numpy(x)) == x`.
    # (We avoid `np.from_dlpack` here: numpy >= 2.1 imports DLPack buffers as
    # read-only, which would break the documented write-back behaviour.)
    return np.asarray(Py(x)).T
end
