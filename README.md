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
pkg> add 
```

## Usage Examples

```julia
julia> using HuggingFaceDatasets

julia> dataset = load_dataset("mnist", split="train")

julia> dataset[1]
```


## Datasets list

https://huggingface.co/datasets
