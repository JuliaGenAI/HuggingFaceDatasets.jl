# HuggingFaceDatasets

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGenAI.github.io/HuggingFaceDatasets.jl/dev)
[![Build Status](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGenAI/HuggingFaceDatasets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGenAI/HuggingFaceDatasets.jl) 

HuggingFaceDatasets.jl is a non-official julia wrapper around the python package  `datasets` from Hugging Face. `datasets` contains a large collection of machine learning datasets (see [here](https://huggingface.co/datasets) for a list) that this package makes available to the julia ecosystem.

It wraps the python `datasets.Dataset` and `datasets.DatasetDict` types and:

- forwards every method of the underlying python object, so the full `datasets` API (`map`, `filter`, `shuffle`, `train_test_split`, `cast_column`, …) is available;
- uses 1-based indexing, julia iteration, and other julia conventions;
- returns observations in a lazy `"julia"` format by default, converting them to native julia types (numeric N-D arrays, dictionaries, …) on access, copylessly when possible, and stacking array columns into `(dims…, N)` tensors.

This package is built on top of [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl). See the [documentation](https://JuliaGenAI.github.io/HuggingFaceDatasets.jl/dev) for a full guide covering method forwarding, the transform workflow, array orientation and working with images, and integration with MLUtils/Flux data loaders.

## Installation

HuggingFaceDatasets.jl is a registered Julia package. You can easily install it through the package manager:

```julia
pkg> add HuggingFaceDatasets
```

## Usage

HuggingFaceDatasets.jl provides wrappers around types from the `datasets` python package (e.g. `Dataset` and `DatasetDict`) along with a few related methods.

Check out the [examples/](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/tree/main/examples) folder for usage examples.

Observations are returned in the `"julia"` format by default, i.e. converted to native julia types on access:

```julia
julia> using HuggingFaceDatasets

julia> train_data = load_dataset("ylecun/mnist", split = "train")
Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})

julia> length(train_data)
60000

julia> train_data[1]["label"]      # a native julia value
5

julia> train_data[1]["image"]      # image as a raw (W, H) numeric array (see the docs on images)
28×28 Matrix{UInt8}:
[...]

julia> train_data[1:2]["label"]    # a batch: each column becomes a vector
2-element Vector{Int64}:
 5
 0
```

Pass `set_format!(train_data, nothing)` (or use the underlying `.py` object) to opt out and get the raw Python observations instead.

## Troubleshooting

- If having problems in resolving the CondaPkg environment, try to set `ENV["JULIA_CONDAPKG_OPENSSL_VERSION"] = true` before loading the package. See more details [here](https://github.com/JuliaPy/CondaPkg.jl?tab=readme-ov-file#preferences)
