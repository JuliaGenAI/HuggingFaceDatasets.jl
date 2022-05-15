module HuggingFaceDatasets

using Base: @kwdef
using PythonCall
using MLUtils: getobs, numobs
import MLUtils

export datasets, set_transform!, 
      Dataset, load_dataset

include("observation.jl")
include("dataset.jl")

include("transforms.jl")
export py2jl

const datasets = PythonCall.pynew()

PythonCall.pyconvert(x) = pyconvert(Any, x)

function load_dataset(args...; kws...)
    d = datasets.load_dataset(args...; kws...)
    if pyisinstance(d, datasets.Dataset)
        return Dataset(d)
    else
        return d
    end
end

function __init__()
    PythonCall.pycopy!(datasets, pyimport("datasets"))
end

end # module
