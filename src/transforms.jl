
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

# PythonCall's wrappers convert like the Python object they wrap.
py2jl(x::Union{PyArray, PyDict, PyList, PySet}) = py2jl(Py(x))

# Whether a numpy array's dtype is one `numpy2jl` can share zero-copy: bool,
# signed/unsigned integer, float, or complex. Strings/objects/datetimes are excluded.
_is_numeric_dtype(x::Py) = pyconvert(String, x.dtype.kind) in ("b", "i", "u", "f", "c")

function py2jl(x::Py)
    # Strings first: a text batch converts one per row, and each check below costs a Python call.
    if pyisinstance(x, pybuiltins.str)
        return pyconvert(String, x)
    # handle datasets
    elseif pyisinstance(x, datasets.Dataset)
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
        # plain scalar cell (e.g. a `bool` cell, or the output of a `map` returning a Julia/Python
        # scalar) to a 0-d `ndarray`. Unwrap it to a numpy scalar of the same dtype (`x[()]`) and
        # convert that, so a scalar reads back as a scalar rather than a `fill(x)` 0-d array.
        if pyconvert(Int, x.ndim) == 0
            return py2jl(x[()])
        # Zero-copy sharing (`numpy2jl`) only supports numeric dtypes. Non-numeric arrays (strings,
        # `object` arrays from ragged columns, datetimes, ...) fall back to a nested-list
        # conversion, so a string column still comes back as a `Vector{String}`.
        elseif _is_numeric_dtype(x)
            return numpy2jl(x)
        # A list column of numbers from the "julia" formatter, which also holds the rows as one
        # flat buffer: two zero-copy conversions instead of one per row.
        elseif pyisinstance(x, pyhfds.RaggedArray) && !pyis(x.values, pybuiltins.None)
            return RaggedVector(numpy2jl(x.values), numpy2jl(x.offsets))
        else
            return py2jl(x.tolist())
        end
    # handle numpy scalars, which the numpy-based formatter yields for single cells. A numeric
    # one becomes the Julia number of the same type (`np.float32` → `Float32`), where `.item()`
    # would widen it to `Int64`/`Float64`.
    elseif pyisinstance(x, np.number)
        v = pyconvert(Any, x)
        # aliases without a PythonCall rule (e.g. `np.longlong`) convert to a 0-d `PyArray`
        return v isa AbstractArray ? v[] : v
    # other numpy scalars (`np.bool_`, `np.datetime64`, ...): `.item()` gives the Python scalar
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


# Roots the numpy array behind each zero-copy `numpy2jl` result for as long as Julia memory
# views it. Entries are matched by identity, never by content (#106), and owned by the array's
# `Memory` on Julia ≥ 1.11, which `reshape`/`vec` of the array share without referencing the
# array itself. As in a `WeakKeyDict`, the owner's finalizer only marks the table dirty, and the
# next `numpy2jl` call drops the dead entries, releasing their numpy arrays then and there.
#
# The GC doesn't count the wrapped memory, so a loop whose only allocations are such arrays would
# keep every one alive (#107). Once the rooted bytes reach twice what was live after the last
# collection that `numpy2jl` ran (and at least `_NUMPY_GC_BYTES`), it runs a young collection,
# and a full one if that doesn't free half of them.
mutable struct NumpyRoots
    const lock::ReentrantLock
    const owners::Vector{WeakRef}
    const arrays::Vector{Py}
    const sizes::Vector{Int}
    bytes::Int
    gc_bytes::Int
    @atomic dirty::Bool
end

const _NUMPY_GC_BYTES = 256 * 2^20

const _NUMPY_ROOTS = NumpyRoots(ReentrantLock(), WeakRef[], Py[], Int[], 0, _NUMPY_GC_BYTES, false)

_buffer_owner(a::Array) = @static VERSION >= v"1.11" ? a.ref.mem : a

_mark_dirty(_) = (@atomic _NUMPY_ROOTS.dirty = true; nothing)

# Drops the entries whose owner was collected, returning their numpy arrays for the caller to
# release outside the lock. Call with `r.lock` held. `force` scans even if no owner's finalizer
# ran, as when the caller of a collection holds a lock, which disables finalizers.
function _purge!(r::NumpyRoots; force::Bool = false)
    dead = Py[]
    force || (@atomic r.dirty) || return dead
    @atomic r.dirty = false
    j = 0
    for i in eachindex(r.owners)
        if r.owners[i].value === nothing
            push!(dead, r.arrays[i])
            r.bytes -= r.sizes[i]
        else
            j += 1
            r.owners[j], r.arrays[j], r.sizes[j] = r.owners[i], r.arrays[i], r.sizes[i]
        end
    end
    resize!(r.owners, j); resize!(r.arrays, j); resize!(r.sizes, j)
    # Only lowered here: after a young collection, `bytes` can include arrays that survived an
    # earlier one and are dead, which only a full collection frees.
    r.gc_bytes = min(r.gc_bytes, max(_NUMPY_GC_BYTES, 2 * r.bytes))
    return dead
end

function _purge_and_release!(r::NumpyRoots)
    foreach(PythonCall.pydel!, @lock r.lock _purge!(r; force = true))
end

function _root!(owner, x::Py, nbytes::Int)
    r = _NUMPY_ROOTS
    finalizer(_mark_dirty, owner)
    dead, over = @lock r.lock begin
        dead = _purge!(r)
        push!(r.owners, WeakRef(owner))
        push!(r.arrays, PythonCall.pynew(x))
        push!(r.sizes, nbytes)
        r.bytes += nbytes
        dead, r.bytes > r.gc_bytes
    end
    foreach(PythonCall.pydel!, dead)
    over && _collect_numpy!(r)
    return nothing
end

function _collect_numpy!(r::NumpyRoots)
    trigger = r.gc_bytes
    GC.gc(false)
    _purge_and_release!(r)
    if r.bytes > trigger ÷ 2
        GC.gc(true)
        _purge_and_release!(r)
    end
    @lock r.lock r.gc_bytes = max(_NUMPY_GC_BYTES, 2 * r.bytes)
    return nothing
end

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
    # `PyArray(x.T)` is a zero-copy view with reversed axes. `unsafe_wrap` then exposes that same
    # memory as a genuine `Array`, so the result stays on Julia's fast paths (BLAS, GPU host→device
    # copies, linear indexing) that a `PyArray` — not being a `DenseArray` — would miss.
    # `_NUMPY_ROOTS` keeps `x` alive while any Julia view of its memory is.
    p = PyArray(x.T)
    arr = unsafe_wrap(Array, pointer(p), size(p); own = false)
    _root!(_buffer_owner(arr), x, sizeof(arr))
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
leaves.

A narrow numeric scalar (`Int8`–`Int32`, `UInt8`–`UInt64`, `Float16`, `Float32`) becomes a
numpy scalar of the same dtype, so `datasets` stores it with its type (a `Float32` becomes
`float32`, not `float64`). `Bool`, `Int64` and `Float64` become Python `bool`, `int` and
`float`.

A vector of numpy-native numbers (`Bool`, `Int8`–`Int64`, `UInt8`–`UInt64`,
`Float16`–`Float64`) becomes a 1-D numpy array of the same dtype, so `datasets` stores it
with its element type (a `Vector{UInt16}` becomes `uint16`, not `int64`). Any other vector
(of strings, arrays, `Union{Missing, Int}`, `BigInt`, ...) becomes a Python list of converted
elements. Multi-dimensional numeric `AbstractArray`s are converted with [`jl2numpy`](@ref)
(with the documented axis reversal); other leaves are handed to PythonCall's default `Py`
conversion.

Unlike `jl2numpy`, which shares memory, `jl2py` copies numeric vectors and multi-dimensional
arrays, so mutating them afterwards does not affect the result.

This is the write-path dual of `py2jl`, used to bridge pure-Julia callbacks into the
Python `datasets` API (see the Julia-friendly [`map`](@ref) / [`filter`](@ref) overloads).

# Examples

```jldoctest
julia> jl2py(Dict("label" => [1, 2, 3]))
Python: {'label': array([1, 2, 3])}

julia> jl2py(UInt16[1, 2, 3])
Python: array([1, 2, 3], dtype=uint16)

julia> jl2py(Float32(1.5))
Python: np.float32(1.5)

julia> jl2py((1, "a", ["b", "c"]))
Python: (1, 'a', ['b', 'c'])
```
"""
jl2py(x) = Py(x)
jl2py(x::Py) = x

# Numpy scalars, since `datasets` stores a Python `int`/`float` as int64/float64. `Bool`, `Int64`
# and `Float64` already have those dtypes, so they stay Python scalars, which some APIs require
# (`json`, `ClassLabel.int2str`, ...).
for T in (Int8, Int16, Int32, UInt8, UInt16, UInt32, UInt64, Float16, Float32)
    @eval jl2py(x::$T) = np.$(Symbol(lowercase(string(T))))(x)
end

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

# Numpy dtypes that Arrow can store. Complex has no Arrow type; `Int128`, `Rational`, ... would
# become numpy `object` arrays of Julia values.
const _NumpyReal = Union{Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                         Float16, Float32, Float64}

function jl2py(x::AbstractVector)
    # `isconcretetype` rules out `Union{}` and eltypes like `Union{Int32, Int64}` (also `object`)
    if isconcretetype(eltype(x)) && eltype(x) <: _NumpyReal
        # Copy: `datasets` buffers non-batched `map` rows and `IterableDataset` yields them
        # as-is, so a shared view would alias a vector the caller may reuse.
        return jl2numpy(collect(x))
    end
    return pylist(jl2py(el) for el in x)
end

# N-D (N ≥ 2) arrays are copied for the same reason as numeric vectors; non-numeric vectors are
# handled element-wise above so a `Vector` of arrays becomes a Python list of numpy arrays.
jl2py(x::AbstractArray) = jl2numpy(collect(x))
