"""
    Dataset

A Julia wrapper around an object of the python `datasets.Dataset` class.

Provides:
- 1-based indexing.
- All python class' methods from  `datasets.Dataset`.

Usually constructed via [`load_dataset`](@ref) or from in-memory Julia data (see below),
both of which default to the `"julia"` format so observations are converted to native Julia
types on access. A raw `datasets.Dataset` object can also be wrapped directly, in which case
its current Python format is preserved (use [`with_format`](@ref) to opt in to `"julia"`).

See also [`load_dataset`](@ref), [`DatasetDict`](@ref), [`with_format`](@ref), and
[`reset_format!`](@ref).

# Examples

```jldoctest
julia> ds = Dataset((; label=[5, 0, 4]))
Dataset({
    features: ['label'],
    num_rows: 3
})

julia> length(ds)
3

julia> ds[1]      # observations are Julia values by default (the "julia" format)
Dict{String, Int64} with 1 entry:
  "label" => 5

julia> ds[1:3]    # a range or vector returns a batch (columns -> vectors)
Dict{String, Vector{Int64}} with 1 entry:
  "label" => [5, 0, 4]

julia> set_format!(ds, nothing);   # opt out: hand back the raw Python observations

julia> ds[1]
Python: {'label': 5}
```
"""
mutable struct Dataset
    py::Py
    jltransform

    function Dataset(py::Py, jltransform = identity)
        if !pyisinstance(py, datasets.Dataset)
            throw(ArgumentError("expected a `datasets.Dataset`, got $(pytype(py))"))
        end
        return new(py, jltransform)
    end
end

"""
    Dataset(d::AbstractDict; jltransform = nothing)
    Dataset(nt::NamedTuple; jltransform = nothing)

Build a `Dataset` from in-memory Julia data: a `Dict` or `NamedTuple` mapping column
names to columns, delegating to `datasets.Dataset.from_dict`.

Each column is either

- a vector of scalars (numbers, strings, booleans, ...), or
- an **N-D array** column, given either as a vector-of-arrays (one array per observation,
  e.g. a `Vector{Matrix}` image column) or as a single stacked `(dims…, N)` array whose
  **last axis** indexes the `N` observations (the MLUtils convention).

Array columns are converted with [`jl2numpy`](@ref), so the column-major/row-major axis
reversal is handled and round-trips: a single row reads back as the original Julia array
and a range index stacks rows into a `(dims…, N)` tensor.

The dataset is returned in the `"julia"` format by default (observations converted to
native Julia types on access). Pass a `jltransform` to install a custom transform instead,
or call [`reset_format!`](@ref) for the raw Python observations.

See also [`with_format`](@ref), [`jl2py`](@ref), and [`jl2numpy`](@ref).

# Examples

```jldoctest
julia> ds = Dataset((label = [5, 0, 4], text = ["a", "b", "c"]))
Dataset({
    features: ['label', 'text'],
    num_rows: 3
})

julia> ds[1:3]["label"]
3-element Vector{Int64}:
 5
 0
 4

julia> ds = Dataset((; x = [1 2 3; 4 5 6]));   # a 2×3 matrix column: 3 observations

julia> ds[1]["x"]           # observation 1 is the first column of the input
2-element Vector{Int64}:
 1
 4

julia> ds[1:3]["x"]         # a range stacks observations along the last axis
2×3 Matrix{Int64}:
 1  2  3
 4  5  6
```
"""
Dataset(d::AbstractDict; jltransform = nothing) =
    _from_julia_data(_from_dict_pydict(d), jltransform)

Dataset(nt::NamedTuple; jltransform = nothing) =
    _from_julia_data(_from_dict_pydict(nt), jltransform)

# Wrap a freshly built `datasets.Dataset` from Julia data. With no explicit `jltransform`
# the dataset defaults to the `"julia"` format; passing a transform installs it instead
# (leaving the Python format untouched, i.e. observations arrive as raw Python batches).
function _from_julia_data(py::Py, jltransform)
    ds = Dataset(datasets.Dataset.from_dict(py))
    return jltransform === nothing ? set_format!(ds, "julia") : set_jltransform!(ds, jltransform)
end

# Convert a Julia column mapping (Dict/NamedTuple) into a Python dict suitable for
# `datasets.Dataset.from_dict`.
function _from_dict_pydict(d)
    py = pydict()
    for (k, v) in pairs(d)
        py[string(k)] = _column_to_py(string(k), v)
    end
    return py
end

# A column given as a vector: each element is one observation. `jl2py` maps a scalar
# vector to a Python list, and a vector-of-arrays to a Python list of (axis-reversed)
# numpy arrays, which `from_dict` infers as `List(List(...))`.
_column_to_py(k::AbstractString, v::AbstractVector) = jl2py(v)

# A column given as a single stacked N-D array: the LAST axis indexes observations (the
# MLUtils convention, matching what the numpy read path stacks into), so split along it
# into per-observation arrays before handing off to `jl2py`.
function _column_to_py(k::AbstractString, v::AbstractArray)
    n = size(v, ndims(v))
    obs = [copy(selectdim(v, ndims(v), i)) for i in 1:n]
    return jl2py(obs)
end

_column_to_py(k::AbstractString, v) = throw(ArgumentError(
    "column \"$k\" must be an AbstractArray of observations, got $(typeof(v))"))

function Base.getproperty(ds::Dataset, s::Symbol)
    if s in fieldnames(Dataset)
        return getfield(ds, s)
    elseif s === :with_format
        return format -> with_format(ds, format)
    else
        res = getproperty(getfield(ds, :py), s)
        if pycallable(res)
            return CallableWrapper(res, getfield(ds, :jltransform))
        else
            return res |> py2jl
        end
    end
end

Base.length(ds::Dataset) = length(ds.py)

Base.firstindex(ds::Dataset) = 1
Base.lastindex(ds::Dataset) = length(ds)

# Iterate over observations, so that `for obs in ds`, `collect(ds)`, etc. work.
function Base.iterate(ds::Dataset, state=(1, length(ds)))
    i, n = state
    i > n && return nothing
    return ds[i], (i + 1, n)
end

Base.getindex(ds::Dataset, ::Colon) = ds[1:length(ds)]

function Base.getindex(ds::Dataset, i::AbstractVector{<:Integer})
    all(≥(1), i) || throw(BoundsError(ds, i))
    x = getfield(ds, :py)[i .- 1]
    return ds.jltransform(x)
end

function Base.getindex(ds::Dataset, i::Integer)
    x = ds[[i]] # transforms and jltransforms always work on batches
    return getobs(x, 1)
end

function Base.getindex(ds::Dataset, i::AbstractString)
    x = ds.py[i]
    d = @py {i: x}
    return ds.jltransform(d)[i]
end

"""
    map(f, ds::Dataset; kws...)

Apply `f` to `ds` through `datasets`' `map`, bridging Julia values on both sides: each
example (or batch, with `batched=true`) is converted with [`py2jl`](@ref) before `f` sees
it, and `f`'s return value is converted back to Python with [`jl2py`](@ref). This lets you
write pure-Julia transforms while still getting `datasets`' batching, caching, and
multiprocessing.

Keyword arguments (`batched`, `num_proc`, `remove_columns`, ...) are forwarded to the
Python `map`. The parent's julia format/transform is preserved on the returned `Dataset`.

Use `ds.map(...)` (the forwarded Python method) if you need to hand `map` a raw Python
callback instead.

See also [`filter`](@ref).

# Examples

```julia
julia> ds = with_format(Dataset((; label=[5, 0, 4])), "julia");

julia> ds2 = map(x -> Dict("label" => x["label"] .+ 100), ds; batched=true);

julia> ds2[1:3]["label"]
3-element Vector{Int64}:
 105
 100
 104
```
"""
function Base.map(f, ds::Dataset; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(ds, :py).map(g; kws...)
    return Dataset(y, getfield(ds, :jltransform))
end

"""
    filter(f, ds::Dataset; kws...)

Filter `ds` through `datasets`' `filter`, bridging Julia values: each example (or batch,
with `batched=true`) is converted with [`py2jl`](@ref) before `f` sees it, and `f` returns
a `Bool` (or, when `batched=true`, a `Vector{Bool}`), converted back to Python with
[`jl2py`](@ref).

Keyword arguments are forwarded to the Python `filter`; the parent's julia format/transform
is preserved on the returned `Dataset`.

See also [`map`](@ref).
"""
function Base.filter(f, ds::Dataset; kws...)
    g = x -> jl2py(f(py2jl(x)))
    y = getfield(ds, :py).filter(g; kws...)
    return Dataset(y, getfield(ds, :jltransform))
end

function Base.deepcopy_internal(ds::Dataset, stackdict::IdDict)
    haskey(stackdict, ds) && return stackdict[ds]::Dataset
    py = pycopy.deepcopy(getfield(ds, :py))
    jltransform = Base.deepcopy_internal(getfield(ds, :jltransform), stackdict)
    ds2 = Dataset(py, jltransform)
    stackdict[ds] = ds2
    return ds2
end

# Shallow copy: the returned dataset shares the underlying Arrow data with `ds`
# but has an independent format, so `set_format!`/`set_jltransform!` on the copy
# do not affect the original. Used internally by the copy-on-write helpers.
function Base.copy(ds::Dataset)
    py = pycopy.copy(getfield(ds, :py))
    return Dataset(py, getfield(ds, :jltransform))
end

Base.show(io::IO, ds::Dataset) = print(io, ds.py)

"""
    with_format(ds::Dataset, format)

Return a copy of `ds` with the format set to `format`.
If format is `"julia"`, the returned dataset is backed by `datasets`' `numpy` format and
transformed with [`py2jl`](@ref), using copyless conversion from python types when possible.
Any other string is forwarded to `datasets`' own `set_format` (`"numpy"`, `"torch"`, ...),
with observations left as raw Python objects.

See also [`set_format!`](@ref) and [`reset_format!`](@ref).

# Examples

```jldoctest
julia> ds = set_format!(Dataset((; label=[5, 0, 4])), nothing);   # start from raw Python

julia> ds[1]
Python: {'label': 5}

julia> ds = with_format(ds, "julia");

julia> ds[1]
Dict{String, Int64} with 1 entry:
  "label" => 5
```
"""
function with_format(ds::Dataset, format::AbstractString)
    ds = copy(ds)
    return set_format!(ds, format)
end

"""
    set_format!(ds::Dataset, format)

Set the format of `ds` to `format`. Mutating version of [`with_format`](@ref).

`format == "julia"` installs the julia format (numpy-backed + [`py2jl`](@ref)); `nothing`
removes all formatting (raw Python observations); any other string is forwarded to
`datasets`' `set_format` (`"numpy"`, `"torch"`, ...). The single-argument form
`set_format!(ds)` restores the default julia format (see [`reset_format!`](@ref)).
"""
function set_format!(ds::Dataset, format)
    if format == "julia"
        # Use the numpy format so numeric array columns decode as real N-D arrays and a
        # range index stacks rows into an `(N, dims…)` tensor; `py2jl` (via DLPack) then
        # reverses the axes to a Julia `(dims…, N)` array with the observation axis last.
        ds.py.set_format("numpy")
        ds.jltransform = py2jl
    else
        ds.py.set_format(format)
        ds.jltransform = identity
    end
    return ds
end

set_format!(ds::Dataset) = reset_format!(ds)

"""
    reset_format!(ds::Dataset)

Reset `ds` to the default `"julia"` format, i.e. `set_format!(ds, "julia")`. To instead
strip all formatting and get the raw Python observations, use `set_format!(ds, nothing)`.
"""
reset_format!(ds::Dataset) = set_format!(ds, "julia")

"""
    with_jltransform(ds::Dataset, transform)
    with_jltransform(transform, ds::Dataset)

Return a copy of `ds` with the julia transform set to `transform`.
The `transform` applies when indexing, e.g. `ds[1]` or `ds[1:2]`.

The transform is always applied to a batch of data, even if the index is a single integer.
That is, `ds[1]` is equivalent to `ds[1:1]` from the point of view of the transform.

The julia transform is applied after the python transform (if any). 
The python transform can be set with `ds.set_transform(pytransform)`.

If `transform` is `nothing` or `identity`, the returned dataset will not be transformed.

See also [`set_jltransform!`](@ref) for the mutating version.
"""
function with_jltransform(ds::Dataset, transform)
    ds = copy(ds)
    return set_jltransform!(ds, transform)
end

# conveniency for the do syntax
with_jltransform(transform, ds::Dataset) = with_jltransform(ds, transform)

"""
    set_jltransform!(ds::Dataset, transform)
    set_jltransform!(transform, ds::Dataset)

Set the julia transform of `ds` to `transform`. Mutating
version of [`with_jltransform`](@ref).
"""
function set_jltransform!(ds::Dataset, transform)
    if transform === nothing
        ds.jltransform = identity
    else
        ds.jltransform = transform
    end
    return ds
end

set_jltransform!(transform, ds::Dataset) = set_jltransform!(ds, transform)
