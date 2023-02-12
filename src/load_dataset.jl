
"""
    load_dataset(args...; kws...)

Load a dataset from the [HuggingFace Datasets](https://huggingface.co/datasets) library.

All arguments are passed to the python function `datasets.load_dataset`.
See the documentation [here](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset).

Returns a [`DatasetDict`](@ref) or a [`Dataset`](@ref) depending on the `split` argument.

Use the `dataset.with_format("julia")` to lazily convert the observation from the dataset 
to julia types.

# Examples
Without a `split` argument, a `DatasetDict` is returned:

```julia
julia> d = load_dataset("glue", "sst2")
DatasetDict(<py DatasetDict({
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
})>, identity)

julia> d["train"]
Dataset(<py Dataset({
    features: ['sentence', 'label', 'idx'],
    num_rows: 67349
})>, identity)
```

Selecting a split returns a `Dataset`:

```julia

julia> mnist = load_dataset("mnist", split="train")
Dataset(<py Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})>, identity)

julia> mnist[1]
Python dict: {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x2AE6B2B30>, 'label': 5}
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

