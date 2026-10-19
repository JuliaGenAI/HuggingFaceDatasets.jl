module HuggingFaceDatasets

using PythonCall
using Compat: @compat
using MLCore: getobs
using ImageCore: colorview, RGB, Gray, N0f8
using Tables: Tables
using SHA: SHA

const datasets = PythonCall.pynew()
const PIL = PythonCall.pynew()
const np = PythonCall.pynew()
const pycopy = PythonCall.pynew() # the python `copy` module (renamed to avoid shadowing `Base.copy`)
const pickle = PythonCall.pynew() # used to (de)serialize `Dataset` by reference for `Distributed`
const pyhfds = PythonCall.pynew() # this package's Python module, `python/huggingfacedatasets_jl.py`

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

# Declare the wrapper types as Python wrappers (PythonCall's "custom wrappers"): passing one to
# any Python call, as an argument, a keyword (`features = ds.features`) or a container element,
# hands Python the wrapped object instead of an opaque `juliacall.AnyValue`.
for T in (Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Column, Features,
          ClassLabel, Value)
    @eval PythonCall.ispy(::$T) = true
    @eval PythonCall.Py(x::$T) = getfield(x, :py)
end

include("load_dataset.jl")
export load_dataset

include("toplevel.jl")
export concatenate_datasets,
    interleave_datasets,
    load_from_disk

include("serialization.jl")

include("fingerprint.jl")

# `public` is a Julia 1.11+ keyword; `@compat` makes it a no-op on the supported 1.10.
@compat public from_csv, from_json, from_parquet

# Julia < 1.12 bundles libssh2 1.11.0 and loads it before Python starts (at startup on 1.10, with
# `Pkg` on 1.11). On Windows, the conda-forge `libcurl.dll` that `pyarrow` links then binds to that
# already-loaded `libssh2.dll` and misses a symbol from the newer libssh2 it was built against (#29).
function _libssh2_conflict_hint(err)
    Sys.iswindows() && VERSION < v"1.12" && err isa PyException &&
        occursin("DLL load failed", sprint(showerror, err)) || return
    @error """
    Importing the Python `datasets` package failed because a DLL could not be loaded. On Windows \
    with Julia < 1.12 this is a known conflict: the conda-forge `libcurl` used by `pyarrow` needs \
    a newer `libssh2` than the one bundled with Julia, which is already loaded. Either use \
    Julia ≥ 1.12, or cap the conda `libssh2` at Julia's version and restart Julia:

        using CondaPkg; CondaPkg.add("libssh2"; version="<1.11.1")

    See https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/issues/29.
    """
end

function __init__()
    # Since it is illegal in PythonCall to import a python module in a module, we need to do this here.
    # https://juliapy.github.io/PythonCall.jl/dev/pythoncall-reference/#PythonCall.Core.pycopy!
    try
        PythonCall.pycopy!(datasets, pyimport("datasets"))
    catch err
        _libssh2_conflict_hint(err)
        rethrow()
    end
    PythonCall.pycopy!(PIL, pyimport("PIL"))
    pyimport("PIL.Image")
    pyimport("PIL.PngImagePlugin")
    pyimport("PIL.JpegImagePlugin")
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(pycopy, pyimport("copy"))
    PythonCall.pycopy!(pickle, pyimport("pickle"))

    # Register the "julia" format type with `datasets` by importing its module from a `sys.path`
    # entry, which child interpreters (e.g. spawned `num_proc` workers) inherit, so they can
    # import it too when they unpickle a "julia"-formatted dataset.
    sys = pyimport("sys")
    pydir = pkgdir(@__MODULE__, "src", "python")
    pydir in sys.path || sys.path.append(pydir)
    PythonCall.pycopy!(pyhfds, pyimport("huggingfacedatasets_jl"))

    # Pin the already-resolved Python interpreter for any child processes.
    #
    # The recommended way to parallelize loading past the CPython GIL is
    # `DataLoader(dataset; num_workers=N)`, which spawns `N` `Distributed` worker
    # processes that each `using PythonCall`. Without the following, every worker independently
    # *re-resolves* the CondaPkg environment on startup, in lockstep: they serialize on
    # CondaPkg's file lock and redo work this parent process already did.
    if PythonCall.C.CTX.which === :CondaPkg
        get!(ENV, "JULIA_PYTHONCALL_EXE", PythonCall.python_executable_path()::String)
    end
end

end # module
