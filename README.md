# HuggingFaceDatasets
<!-- 
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://CarloLucibello.github.io/HuggingFaceDatasets.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://CarloLucibello.github.io/HuggingFaceDatasets.jl/dev)
[![Build Status](https://github.com/CarloLucibello/HuggingFaceDatasets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/CarloLucibello/HuggingFaceDatasets.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/CarloLucibello/HuggingFaceDatasets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/CarloLucibello/HuggingFaceDatasets.jl) 
-->

A julia wrapper around the Hugging Face `datasets` python package, exposing a large collection
of machine learning datasets. 

This package is built on top of [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl).

## Installation

This package is under development and not registered yet.

```julia
pkg> add https://github.com/CarloLucibello/HuggingFaceDatasets.jl
```

## Usage Examples

### `load_dataset`

```julia
julia> using HuggingFaceDatasets

julia> train_data = load_dataset("mnist", split="train")
Reusing dataset mnist (/home/carlo/.cache/huggingface/datasets/mnist/mnist/1.0.0 fda16c03c4ecfb13f165ba7e29cf38129ce035011519968cdaf74894ce91c9d4)
Dataset(<py Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})>, HuggingFaceDatasets.py2jl)

julia> train_data[1]
Dict{String, Any} with 2 entries:
  "label" => 5
  "image" => UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
```

### `set_transform!`

```julia
julia> using HuggingFaceDatasets, Flux


julia> train_data = load_dataset("mnist", split="train");

julia> function mnist_transform(x)
            x = py2jl(x) # `py2jl` converts python types to julia types. This is the default transform.
            image = Flux.batch(x["image"]) ./ 255f0
            label = Flux.onehotbatch(x["label"], 0:9)
            return (; image, label)
        end

julia> set_transform!(train_data, mnist_transform)

julia> train_data[1:5].image |> summary
"28×28×5 Array{Float32, 3}"

julia> train_data[1:5].label
10×5 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅
 1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1
```

## Datasets list

For a list of the available datasets, see https://huggingface.co/datasets.
