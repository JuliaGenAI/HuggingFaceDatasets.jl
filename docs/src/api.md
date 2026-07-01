```@meta
CurrentModule = HuggingFaceDatasets
CollapsedDocStrings = true
```

# API Reference

This page documents the public types and functions. See the [Guide](@ref) for
worked examples and background on the transform pipeline.

## Types

```@docs
Dataset
DatasetDict
Column
```

## Loading

```@docs
load_dataset
```

## Formats and transforms

```@docs
with_format
set_format!
reset_format!
with_jltransform
set_jltransform!
```

## Transforming

```@docs
map(f, ::Dataset)
filter(f, ::Dataset)
```

## Type conversion

```@docs
py2jl
jl2py
numpy2jl
jl2numpy
```

## Index

```@index
```
</content>
