```@meta
CurrentModule = HuggingFaceDatasets
DocTestSetup = quote
    using HuggingFaceDatasets, PythonCall
end
```

# Guide

This guide covers how the wrapper relates to the underlying Python `datasets` library,
the `"julia"` format and the transform pipeline, the array/image orientation caveat, and
how to feed a dataset into a Julia data loader.

The examples below build small datasets in memory with
`datasets.Dataset.from_dict` so that they are self-contained and reproducible. In
practice you will usually obtain a dataset from the Hub with [`load_dataset`](@ref), e.g.
`load_dataset("ylecun/mnist", split="train")`; everything shown here applies equally to
those datasets.

## Loading datasets

[`load_dataset`](@ref) forwards all its arguments to the Python
[`datasets.load_dataset`](https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset)
function. What it returns depends on the `split` argument:

- without `split`, you get a [`DatasetDict`](@ref) (a dictionary of splits), e.g.
  `load_dataset("nyu-mll/glue", "sst2")`;
- with a `split`, you get a single [`Dataset`](@ref), e.g.
  `load_dataset("nyu-mll/glue", "sst2", split="train")`.

A [`DatasetDict`](@ref) is an `AbstractDict{String, Dataset}`, so `keys`, `values`,
`haskey`, `get`, and iteration all work as expected:

```jldoctest guide
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
```

A [`Dataset`](@ref) supports 1-based indexing, `length`, `firstindex`/`lastindex`
(so `ds[begin]` and `ds[end]` work), and iteration over observations:

```jldoctest guide
julia> ds = Dataset(datasets.Dataset.from_dict(pydict(label=[5, 0, 4])));

julia> length(ds)
3

julia> ds[begin]
Python: {'label': 5}

julia> ds[end]
Python: {'label': 4}

julia> [obs for obs in ds]      # iteration yields one observation at a time
3-element Vector{Py}:
 {'label': 5}
 {'label': 0}
 {'label': 4}
```

Indexing with a single integer returns one observation; indexing with a range or vector
returns a batch, represented as a dictionary mapping each column name to a vector of
values (see below).

## Method forwarding

Any method or property of the wrapped Python object is available directly on the Julia
wrapper. Method calls are forwarded to Python, and their results are converted back with
[`py2jl`](@ref). This means the whole `datasets` API is usable without a dedicated Julia
binding for each method:

```jldoctest guide
julia> ds = Dataset(datasets.Dataset.from_dict(pydict(label=[0, 1, 2, 3])));

julia> ds.select(0:1)                       # keep the first two rows
Dataset({
    features: ['label'],
    num_rows: 2
})

julia> ds.train_test_split(test_size=0.5, seed=42)
DatasetDict({
    train: Dataset({
        features: ['label'],
        num_rows: 2
    })
    test: Dataset({
        features: ['label'],
        num_rows: 2
    })
})
```

Keyword arguments are forwarded as Python keyword arguments, so calls like
`train_test_split(test_size=0.5)` and `shuffle(seed=0)` behave exactly as in Python.

!!! note "0-based method arguments"
    Methods forwarded to Python keep Python semantics, including **0-based indices**
    (e.g. the argument to `select` above). Only the wrapper's own `getindex`/iteration
    interface is 1-based. Consult the
    [`datasets` documentation](https://huggingface.co/docs/datasets) for the exact
    meaning of each method's arguments.

## The `"julia"` format and transforms

By default, indexing a dataset returns Python objects. The package supports the notion
of a *format* — mirroring `datasets`' own `set_format`/`with_format` — plus an extra
Julia-side transform.

### Setting a format

[`with_format`](@ref) returns a copy of the dataset with a given format;
[`set_format!`](@ref) mutates it in place; [`reset_format!`](@ref) clears it.

```jldoctest guide
julia> ds = Dataset(datasets.Dataset.from_dict(pydict(label=[5, 0, 4])));

julia> ds[1]
Python: {'label': 5}

julia> ds = with_format(ds, "julia");   # or ds.with_format("julia")

julia> ds[1]
Dict{String, Int64} with 1 entry:
  "label" => 5

julia> ds[1:3]                          # a batch: each column becomes a vector
Dict{String, Vector{Int64}} with 1 entry:
  "label" => [5, 0, 4]
```

The special `"julia"` format sets the Julia transform to [`py2jl`](@ref), which
recursively converts Python containers (lists, tuples, dicts, sets), numpy arrays, and
PIL images into native Julia values, copylessly when possible. Any other format string
(e.g. `"numpy"`, `"torch"`, `"pandas"`) is forwarded to the underlying Python
`set_format`.

### Custom Julia transforms

Beyond the format, you can attach an arbitrary Julia function that runs when indexing,
with [`with_jltransform`](@ref) / [`set_jltransform!`](@ref). The transform receives the
raw Python batch, so convert it with [`py2jl`](@ref) first if you want to work with Julia
types:

```jldoctest guide
julia> ds = Dataset(datasets.Dataset.from_dict(pydict(x=[1, 2, 3])));

julia> ds = with_jltransform(ds) do batch
           b = py2jl(batch)          # convert the Python batch to Julia types first
           b["x"] .* 10
       end;

julia> ds[1]
10

julia> ds[1:3]
3-element Vector{Int64}:
 10
 20
 30
```

The transform is **always applied to a batch**, even for a single-integer index: `ds[1]`
is treated as `ds[1:1]` from the transform's point of view, and the single observation is
then extracted.

Note that the format and the custom transform share the same slot: setting the `"julia"`
format installs `py2jl` as the transform, and `with_jltransform` then replaces it — which
is why the transform above calls `py2jl` itself. For layering additional per-batch
processing on top of the `"julia"` format, prefer `MLUtils.mapobs` (see below), as in the
[`examples/flux_mnist.jl`](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/blob/main/examples/flux_mnist.jl)
script.

The order of operations when you index is:

1. the Python format transform (if any) is applied by `datasets`;
2. the Julia transform runs on the resulting Python batch (for the `"julia"` format this
   is [`py2jl`](@ref); otherwise your own function);
3. for single-integer indexing, one observation is extracted from the batch.

`with_format`/`with_jltransform` return a shallow copy that shares the underlying Arrow
data with the original but has an independent format, so setting a format or transform on
the copy does not affect the original.

## Array and image orientation

!!! warning "Arrays come back transposed"
    numpy and PIL are **row-major**, Julia is **column-major**. The zero-copy conversion
    ([`numpy2jl`](@ref), via DLPack) therefore returns an array whose **dimensions are
    reversed** relative to the Python side. A numpy image of shape `(H, W)` becomes a
    Julia array of size `(W, H)`, and an `(H, W, C)` image becomes `(C, W, H)` with the
    channel axis first.

For grayscale and RGB images, [`py2jl`](@ref) already wraps the raw array in an
`ImageCore` view (`Gray{N0f8}` or `RGB{N0f8}`), so you get a proper image type — but note
the spatial axes are still transposed relative to the Python image. Image modes other
than RGB and grayscale (RGBA, CMYK, palette, …) are returned as the raw permuted array.

To recover the original orientation, reverse the axes with `permutedims`:

```jldoctest guide
julia> a = numpy2jl(jl2numpy([1 2 3; 4 5 6]));   # a Julia array, transposed vs. numpy

julia> size(a)
(2, 3)

julia> size(permutedims(a, reverse(1:ndims(a))))
(3, 2)
```

For an image view, use `ImageCore.channelview` to get the underlying numeric array and
permute as needed. Which orientation you want depends on downstream use; for batching
into a Flux model you typically stack `channelview` outputs with the layout your model
expects (see the MNIST example below).

The reverse conversion [`jl2numpy`](@ref) is symmetric: it shares memory through the
buffer protocol and reverses the axes, so `numpy2jl(jl2numpy(x)) == x` holds and writes
propagate in both directions.

## Integration with MLUtils and data loaders

A [`Dataset`](@ref) implements the length/`getindex` interface expected by
[MLUtils.jl](https://github.com/JuliaML/MLUtils.jl), so `numobs`, `getobs`, `mapobs`, and
data loaders such as `Flux.DataLoader` work directly:

```jldoctest guide
julia> using MLUtils

julia> ds = Dataset(datasets.Dataset.from_dict(pydict(x=[1, 2, 3, 4]))).with_format("julia");

julia> numobs(ds)
4

julia> getobs(ds, 2)
Dict{String, Int64} with 1 entry:
  "x" => 2

julia> mapped = mapobs(batch -> batch["x"] .* 10, ds);   # lazily transform observations

julia> getobs(mapped, 1:4)
4-element Vector{Int64}:
 10
 20
 30
 40
```

Putting it together for MNIST, you would set the `"julia"` format, `mapobs` a transform
that turns each image batch into a numeric array (`ImageCore.channelview` +
`Flux.batch`, plus `Flux.onehotbatch` for the labels), then hand the result to a
`Flux.DataLoader`:

```julia-repl
julia> train_data = load_dataset("ylecun/mnist", split="train").with_format("julia");

julia> train_data = mapobs(mnist_transform, train_data)[:];   # lazily map, then materialize

julia> train_loader = Flux.DataLoader(train_data; batchsize=128, shuffle=true);
```

Materializing with `[:]` loads the whole (transformed) dataset into memory, which is
fastest for small datasets like MNIST. Dropping the `[:]` keeps loading on-the-fly, which
is slower per epoch but avoids holding everything in memory.

This snippet needs the Hub and the Flux/ImageCore stack, so it is not run as a doctest; a
complete, runnable version — including `mnist_transform` and the training loop — lives in
[`examples/flux_mnist.jl`](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/blob/main/examples/flux_mnist.jl).
```
</content>
