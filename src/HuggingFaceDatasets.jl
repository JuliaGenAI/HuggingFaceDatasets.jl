module HuggingFaceDatasets

using PythonCall
using MLUtils: getobs, numobs
import MLUtils

export datasets, load_dataset

include("observation.jl")

include("dataset.jl")
export Dataset, set_transform!

include("datasetdict.jl")
export DatasetDict

include("transforms.jl")
export py2jl

const datasets = PythonCall.pynew()
const PIL = PythonCall.pynew()
const np = PythonCall.pynew()

# PYRACY. Remove when https://github.com/cjdoris/PythonCall.jl/issues/172 is closed.
PythonCall.pyconvert(x) = pyconvert(Any, x)

"""
    load_dataset(args...; kws...)

Load a dataset from the [HuggingFace Datasets](https://huggingface.co/datasets) library.

All arguments are passed to the python function `datasets.load_dataset`.

# Examples

```julia
julia> d =load_dataset("glue", "sst2")
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
})>, HuggingFaceDatasets.py2jl)

julia> d["train"]
Dataset(<py Dataset({
    features: ['sentence', 'label', 'idx'],
    num_rows: 67349
})>, HuggingFaceDatasets.py2jl)

mnist = load_dataset("mnist", split="train")

julia> mnist = load_dataset("mnist", split="train")
Dataset(<py Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})>, HuggingFaceDatasets.py2jl)

julia> mnist[1]
Dict{String, Any} with 2 entries:
  "label" => 5
  "image" => UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
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

function __init__()
    # Since it is illegal in PythonCall to import a python module in a module, we need to do this here.
    # https://cjdoris.github.io/PythonCall.jl/dev/pythoncall-reference/#PythonCall.pycopy!
    PythonCall.pycopy!(datasets, pyimport("datasets"))
    PythonCall.pycopy!(PIL, pyimport("PIL"))
    PythonCall.pycopy!(np, pyimport("numpy"))
end

end # module
