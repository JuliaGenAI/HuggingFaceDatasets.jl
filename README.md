# HuggingFaceDatasets

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGenAI.github.io/HuggingFaceDatasets.jl/dev)
[![Build Status](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGenAI/HuggingFaceDatasets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGenAI/HuggingFaceDatasets.jl) 

HuggingFaceDatasets.jl is a non-official julia wrapper around the python package  `datasets` from Hugging Face. `datasets` contains a large collection of machine learning datasets (see [here](https://huggingface.co/datasets) for a list) that this package makes available to the julia ecosystem.

This package is built on top of [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl).

## Installation

HuggingFaceDatasets.jl is a registered Julia package. You can easily install it through the package manager:

```julia
pkg> add HuggingFaceDatasets
```

## Usage

HuggingFaceDatasets.jl provides wrappers around types from the `datasets` python package (e.g. `Dataset` and `DatasetDict`) along with a few related methods.

Check out the [examples/](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/tree/main/examples) folder for usage examples.

```julia
julia> using HuggingFaceDatasets

julia> train_data = load_dataset("mnist", split = "train")
Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})

julia> train_data[1]
Python: {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x3340B0290>, 'label': 5}

julia> length(train_data)
60000

julia> train_data = load_dataset("mnist", split = "train").with_format("julia");

julia> train_data[1] # Returned observations are now julia objects
Dict{String, Any} with 2 entries:
  "label" => 5
  "image" => Gray{N0f8}[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]

julia> train_data[1:2]
Dict{String, Vector} with 2 entries:
  "label" => [5, 0]
  "image" => ReinterpretArray{Gray{N0f8}, 2, UInt8, Matrix{UInt8}, false}[[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0…
```

## Troubleshooting

- If having problems in resolving the CondaPkg environment, try to set `ENV["JULIA_CONDAPKG_OPENSSL_VERSION"] = true`before loading the package. See more details [here](https://github.com/JuliaPy/CondaPkg.jl?tab=readme-ov-file#preferences)
