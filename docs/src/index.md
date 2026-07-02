```@meta
CurrentModule = HuggingFaceDatasets
DocTestSetup = quote
    using HuggingFaceDatasets, PythonCall
end
```

# HuggingFaceDatasets

Documentation for [HuggingFaceDatasets](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl).

HuggingFaceDatasets.jl is a non-official Julia wrapper around the Python package
[`datasets`](https://huggingface.co/docs/datasets) from Hugging Face. `datasets`
provides access to a large collection of machine learning datasets
(see [the Hub](https://huggingface.co/datasets) for the full list), which this
package makes available to the Julia ecosystem.

The package wraps the Python `datasets.Dataset` and `datasets.DatasetDict` types and:

- forwards every method of the underlying Python object, so the full `datasets` API
  (`map`, `filter`, `shuffle`, `train_test_split`, `cast_column`, …) is available;
- uses 1-based indexing, Julia iteration, and other Julia conventions;
- returns observations in a lazy `"julia"` format by default, converting them to native
  Julia types (numeric N-D arrays, dictionaries, ...) on access, copylessly when possible,
  and stacking array columns into `(dims…, N)` tensors.

It is built on top of [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl).

## Installation

HuggingFaceDatasets.jl is a registered Julia package. Install it through the package
manager:

```julia
pkg> add HuggingFaceDatasets
```

The Python `datasets` package and its dependencies are installed automatically through
[CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) the first time the package is
loaded; no manual Python setup is required.

## Quickstart

Fetch a dataset from the Hub with [`load_dataset`](@ref) and index into it. Observations
are returned in the `"julia"` format by default, so they are lazily converted to native
Julia types (dictionaries, numeric arrays, ...) on access:

```julia
julia> using HuggingFaceDatasets

julia> train_data = load_dataset("ylecun/mnist", split = "train")
Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})

julia> train_data[1]["label"]
5

julia> train_data[1]["image"]           # a raw (W, H) numeric array (see the Guide)
28×28 Matrix{UInt8}:
[...]
```

The same applies to an in-memory dataset, which is handy for reproducible examples:

```jldoctest
julia> using HuggingFaceDatasets, PythonCall

julia> ds = Dataset((; label=[5, 0, 4]));

julia> ds[1]                            # native Julia value by default
Dict{String, Int64} with 1 entry:
  "label" => 5

julia> ds[1:3]                          # a batch: each column becomes a vector
Dict{String, Vector{Int64}} with 1 entry:
  "label" => [5, 0, 4]

julia> set_format!(ds, nothing);        # opt out: raw Python observations

julia> ds[1]
Python: {'label': 5}
```

See the [Guide](@ref) for the transform workflow, method forwarding, array/image
orientation, and integration with MLUtils/Flux data loaders. Runnable examples
live in the
[`examples/`](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/tree/main/examples)
folder.

## Troubleshooting

- If you have problems resolving the CondaPkg environment, try setting
  `ENV["JULIA_CONDAPKG_OPENSSL_VERSION"] = true` before loading the package. See more
  details [here](https://github.com/JuliaPy/CondaPkg.jl?tab=readme-ov-file#preferences).
</content>
