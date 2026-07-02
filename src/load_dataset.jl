
"""
    load_dataset(args...; kws...)

Load a dataset from the [HuggingFace Datasets](https://huggingface.co/datasets) library.

All arguments are passed to the python function `datasets.load_dataset`.
See the documentation [here](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset).

Returns a [`DatasetDict`](@ref) or a [`Dataset`](@ref) depending on the `split` argument.
With `streaming=true` it instead returns the lazy [`IterableDatasetDict`](@ref) or
[`IterableDataset`](@ref) counterpart (consumed by iteration, not indexing).

The result is returned in the `"julia"` format, so observations are lazily converted to
native Julia types on access (see [`with_format`](@ref)). Use `set_format!(ds, nothing)`
(or the underlying `.py` object) if you want the raw Python observations instead.

# Examples

Without a `split` argument, a `DatasetDict` is returned:

```julia
julia> d = load_dataset("nyu-mll/glue", "sst2")
DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 67349
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 872
    })
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1821
    })
})

julia> d["train"]
Dataset({
    features: ['sentence', 'label', 'idx'],
    num_rows: 67349
})
```

Selecting a split returns a `Dataset` instead. Observations come back as native Julia
values thanks to the default `"julia"` format:

```julia
julia> mnist = load_dataset("ylecun/mnist", split="train")
Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})

julia> mnist[1]["label"]
5

julia> mnist[1]["image"]        # a raw (W, H) numeric array under the numpy-backed format
28×28 Matrix{UInt8}:
[...]
```
"""
function load_dataset(args...; kws...)
    d = datasets.load_dataset(args...; kws...)
    if pyisinstance(d, datasets.Dataset)
        return set_format!(Dataset(d), "julia")
    elseif pyisinstance(d, datasets.DatasetDict)
        return set_format!(DatasetDict(d), "julia")
    elseif pyisinstance(d, datasets.IterableDataset)
        # `streaming=true` with a `split`: a lazy single-split stream.
        return set_format!(IterableDataset(d), "julia")
    elseif pyisinstance(d, datasets.IterableDatasetDict)
        # `streaming=true` without a `split`: a lazy dict of streams.
        return set_format!(IterableDatasetDict(d), "julia")
    else
        return d
    end
end

