module HuggingFaceDatasets

using PythonCall
using Compat: @compat
using MLCore: getobs
using DLPack: DLPack
using ImageCore: colorview, RGB, Gray, N0f8
using Tables: Tables

const datasets = PythonCall.pynew()
const PIL = PythonCall.pynew()
const np = PythonCall.pynew()
const pycopy = PythonCall.pynew() # the python `copy` module (renamed to avoid shadowing `Base.copy`)
const pickle = PythonCall.pynew() # used to (de)serialize `Dataset` by reference for `Distributed`

export datasets

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

include("iterabledataset.jl")
export IterableDataset

include("iterabledatasetdict.jl")
export IterableDatasetDict

include("column.jl")

include("transforms.jl")
export py2jl,
    jl2py,
    jl2numpy,
    numpy2jl

include("features.jl")
export Features, ClassLabel
@compat public Value, features, class_names, int2str, str2int

include("load_dataset.jl")
export load_dataset

include("toplevel.jl")
export concatenate_datasets,
    interleave_datasets,
    load_from_disk

include("serialization.jl")

# `public` is a Julia 1.11+ keyword; `@compat` makes it a no-op on the supported 1.10.
@compat public from_csv, from_json, from_parquet

function __init__()
    # Since it is illegal in PythonCall to import a python module in a module, we need to do this here.
    # https://juliapy.github.io/PythonCall.jl/dev/pythoncall-reference/#PythonCall.Core.pycopy!
    PythonCall.pycopy!(datasets, pyimport("datasets"))
    PythonCall.pycopy!(PIL, pyimport("PIL"))
    pyimport("PIL.Image")
    pyimport("PIL.PngImagePlugin")
    pyimport("PIL.JpegImagePlugin")
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(pycopy, pyimport("copy"))
    PythonCall.pycopy!(pickle, pyimport("pickle"))
end

end # module
