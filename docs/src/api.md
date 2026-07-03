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
IterableDataset
IterableDatasetDict
Column
```

## Schema (features)

```@docs
features
Features
ClassLabel
Value
class_names
int2str
str2int
```

## Loading

```@docs
load_dataset
load_from_disk
from_csv
from_json
from_parquet
```

## Combining

```@docs
concatenate_datasets
interleave_datasets
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
map(f, ::DatasetDict)
filter(f, ::DatasetDict)
map(f, ::IterableDataset)
filter(f, ::IterableDataset)
map(f, ::IterableDatasetDict)
filter(f, ::IterableDatasetDict)
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
