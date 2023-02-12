module HuggingFaceDatasets

using PythonCall
using MLUtils: getobs, numobs
import MLUtils
using DLPack
using ImageCore

const datasets = PythonCall.pynew()
const PIL = PythonCall.pynew()
const np = PythonCall.pynew()
const copy = PythonCall.pynew()

export datasets

include("observation.jl")

include("callable.jl")

include("dataset.jl")
export Dataset, 
    with_jltransform,
    set_jltransform!, 
    with_format, 
    set_format!,
    reset_format!

include("datasetdict.jl")
export DatasetDict

include("transforms.jl")
export py2jl, 
    jl2numpy, 
    numpy2jl

include("load_dataset.jl")
export load_dataset

function __init__()
    # Since it is illegal in PythonCall to import a python module in a module, we need to do this here.
    # https://cjdoris.github.io/PythonCall.jl/dev/pythoncall-reference/#PythonCall.pycopy!
    PythonCall.pycopy!(datasets, pyimport("datasets"))
    PythonCall.pycopy!(PIL, pyimport("PIL"))
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(copy, pyimport("copy"))
end

end # module
