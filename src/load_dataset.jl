
"""
    load_dataset(args...; kws...)

Load a dataset from the [HuggingFace Datasets](https://huggingface.co/datasets) library.

All arguments are passed to the python function `datasets.load_dataset`.
See the documentation [here](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset).

Returns a [`DatasetDict`](@ref) or a [`Dataset`](@ref) depending on the `split` argument.

Use the `dataset.with_format("julia")` method to lazily convert the observation from the dataset 
to julia types.

# Examples

Without a `split` argument, a `DatasetDict` is returned:

```julia
julia> d = load_dataset("glue", "sst2")
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

Selecting a split returns a `Dataset` instead. We also
apply the `"julia"` format.

```julia
julia> mnist = load_dataset("mnist", split="train").with_format("julia")
Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})

julia> mnist[1]
Dict{String, Any} with 2 entries:
  "label" => 5
  "image" => Gray{N0f8}[Gray{N0f8}(0.0) Gray{N0f8}(0.0) … Gray{N0f8}(0.0) Gray{N0f8}(0.0); Gray{N0f8}(0.0) Gray{N0f8}(0.0) … Gray{N0f…
```
"""
function load_dataset(args...; kws...)
    d = datasets.load_dataset(args...; kws...)
    if pyisinstance(d, datasets.Dataset)
        return Dataset(d)
    elseif pyisinstance(d, datasets.DatasetDict)
        return DatasetDict(d)
    else
        return d
    end
end

