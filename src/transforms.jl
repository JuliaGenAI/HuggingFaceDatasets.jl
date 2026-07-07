
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

# Whether a numpy array's dtype is one `numpy2jl` can share zero-copy: bool,
# signed/unsigned integer, float, or complex. Strings/objects/datetimes are excluded.
_is_numeric_dtype(x::Py) = pyconvert(String, x.dtype.kind) in ("b", "i", "u", "f", "c")

function py2jl(x::Py)
    # handle datasets
    if pyisinstance(x, datasets.Dataset)
        return Dataset(x)
    elseif pyisinstance(x, datasets.DatasetDict)
        return DatasetDict(x)
    elseif pyisinstance(x, datasets.IterableDataset)
        return IterableDataset(x)
    elseif pyisinstance(x, datasets.IterableDatasetDict)
        return IterableDatasetDict(x)
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
        # A 0-d array is a scalar wearing an array's clothes — the numpy formatter tensorizes a
        # plain scalar cell (e.g. the output of a `map` returning a Julia/Python scalar) to a
        # 0-d `ndarray`. Unwrap to a native scalar via `.item()`, mirroring the `np.generic`
        # branch below, so a scalar reads back as a scalar rather than a `fill(x)` 0-d array.
        if pyconvert(Int, x.ndim) == 0
            return py2jl(x.item())
        # Zero-copy sharing (`numpy2jl`) only supports numeric dtypes. Non-numeric arrays (strings,
        # `object` arrays from ragged columns, datetimes, ...) fall back to a nested-list
        # conversion, so a string column still comes back as a `Vector{String}`.
        elseif _is_numeric_dtype(x)
            return numpy2jl(x)
        else
            return py2jl(x.tolist())
        end
    # handle numpy scalars (`np.int64`, `np.float32`, `np.str_`, `np.bool_`, ...), which
    # the numpy format yields for single cells; `.item()` gives the native Python scalar.
    elseif pyisinstance(x, np.generic)
        return py2jl(x.item())
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


# Roots the Python buffer backing each zero-copy `numpy2jl` array for exactly as long as the
# Julia `Array` that views it. The wrapper `Array` is the (weak) key, so an entry — and the
# Python reference it holds — is dropped automatically once that array is garbage-collected;
# `WeakKeyDict` is internally locked, so concurrent inserts/evictions are thread-safe.
#
# Cleanup routes through PythonCall's GIL-deferred decref (a finalizer that can't take the GIL
# just enqueues the pointer), so — unlike DLPack's finalizer, which eagerly re-acquires the GIL
# — a buffer freed on a `DataLoader` worker thread can never deadlock against a thread compiling
# under the GIL.
const _NUMPY_BUFFERS = WeakKeyDict{AbstractArray,Any}()

"""
    numpy2jl(x)

Convert a numpy array to a Julia `Array` sharing memory zero-copy. Mutations to the Julia array
are reflected in the numpy array (and vice versa). Since numpy is row-major and Julia is
column-major, the returned array has permuted (reversed) dimensions.

Read-only or non-contiguous numpy buffers cannot be shared safely and are copied first.

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
    # `unsafe_wrap` below reinterprets the buffer as a column-major contiguous `Array`, which is
    # only valid when `x.T` is F-contiguous, i.e. when `x` is C-contiguous. Copy to a fresh,
    # writable C-contiguous array otherwise. This also covers read-only buffers (numpy >= 2.1
    # marks some columns read-only): aliasing read-only memory with a writable Julia array would
    # be unsound. `x.copy()` defaults to C order, so `x.T` is F-contiguous afterwards.
    if !pyconvert(Bool, x.flags.c_contiguous) || !pyconvert(Bool, x.flags.writeable)
        x = x.copy()
    end
    # `PyArray(x.T)` is a zero-copy view with reversed axes; it holds the Python reference that
    # keeps the buffer alive. `unsafe_wrap` then exposes that same memory as a genuine `Array`, so
    # the result stays on Julia's fast paths (BLAS, GPU host→device copies, linear indexing) that
    # a `PyArray` — not being a `DenseArray` — would miss. `_NUMPY_BUFFERS` roots the `PyArray` for
    # the wrapper's lifetime so the buffer outlives every view of it.
    p = PyArray(x.T)
    arr = unsafe_wrap(Array, pointer(p), size(p); own = false)
    _NUMPY_BUFFERS[arr] = p
    return arr
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

"""
    jl2py(x)

Convert Julia values to Python, the inverse of [`py2jl`](@ref). Recursively traverses
`AbstractDict`, `NamedTuple`, `Tuple`, and `AbstractVector` containers, converting the
leaves. Multi-dimensional numeric `AbstractArray`s are converted with [`jl2numpy`](@ref)
(copyless, with the documented axis reversal); other leaves are handed to PythonCall's
default `Py` conversion.

This is the write-path dual of `py2jl`, used to bridge pure-Julia callbacks into the
Python `datasets` API (see the Julia-friendly [`map`](@ref) / [`filter`](@ref) overloads).

# Examples

```jldoctest
julia> jl2py(Dict("label" => [1, 2, 3]))
Python: {'label': [1, 2, 3]}

julia> jl2py((1, "a", [2, 3]))
Python: (1, 'a', [2, 3])
```
"""
jl2py(x) = Py(x)
jl2py(x::Py) = x

jl2py(x::Dataset)             = getfield(x, :py)
jl2py(x::DatasetDict)         = getfield(x, :py)
jl2py(x::IterableDataset)     = getfield(x, :py)
jl2py(x::IterableDatasetDict) = getfield(x, :py)
jl2py(x::Column)              = getfield(x, :py)
# `jl2py` for the schema views (`Features`/`ClassLabel`/`Value`) is defined in `features.jl`,
# where those types are declared.

function jl2py(x::AbstractDict)
    d = pydict()
    for (k, v) in x
        d[jl2py(k)] = jl2py(v)
    end
    return d
end

function jl2py(x::NamedTuple)
    d = pydict()
    for k in keys(x)
        d[string(k)] = jl2py(x[k])
    end
    return d
end

jl2py(x::Tuple) = pytuple(Tuple(jl2py(el) for el in x))
jl2py(x::AbstractVector) = pylist(jl2py(el) for el in x)
# N-D (N ≥ 2) numeric arrays go through the copyless numpy view; vectors are handled
# element-wise above so a `Vector` of arrays becomes a Python list of numpy arrays.
jl2py(x::AbstractArray) = jl2numpy(x)
