# HuggingFaceDatasets

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://CarloLucibello.github.io/HuggingFaceDatasets.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://CarloLucibello.github.io/HuggingFaceDatasets.jl/dev)
[![Build Status](https://github.com/CarloLucibello/HuggingFaceDatasets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/CarloLucibello/HuggingFaceDatasets.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/CarloLucibello/HuggingFaceDatasets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/CarloLucibello/HuggingFaceDatasets.jl) 

HuggingFaceDatasets.jl is a non-official julia wrapper around the python package  `datasets` from Hugging Face. `datasets` contains a large collection of machine learning datasets (see [here](https://huggingface.co/datasets) for a list) that this package makes available to the julia ecosystem.

This package is built on top of [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl).

## Installation

HuggingFaceDatasets.jl is a registered Julia package. You can easily install it through the package manager:

```julia
pkg> add HuggingFaceDatasets
```

## Usage

HuggingFaceDatasets.jl provides wrappers around types from the `datasets` python package (e.g. `Dataset` and `DatasetDict`) along with a few related methods.

Check out the `examples/` folder for usage examples.

```julia
julia> train_data = load_dataset("mnist", split = "train")
Dataset(<py Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})>, identity)

# Indexing starts with 1. 
# By defaul, python types are returned.
julia> train_data[1]
Python dict: {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x2B64E2E90>, 'label': 5}

julia> set_format!(train_data, "julia")
Dataset(<py Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})>, HuggingFaceDatasets.py2jl)

# Now we have julia types
julia> train_data[1]
Dict{String, Any} with 2 entries:
  "label" => 5
  "image" => UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
```
