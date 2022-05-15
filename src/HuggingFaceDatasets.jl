module HuggingFaceDatasets

using PythonCall
using MLUtils: getobs, numobs
import MLUtils

export datasets, load_dataset

include("observation.jl")

include("dataset.jl")
export Dataset, set_transform!

include("transforms.jl")
export py2jl

const datasets = PythonCall.pynew()

# PYRACY. Remove when https://github.com/cjdoris/PythonCall.jl/issues/172 is closed.
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
