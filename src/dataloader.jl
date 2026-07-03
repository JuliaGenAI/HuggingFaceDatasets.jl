# Transparent integration with `MLUtils.DataLoader` (re-exported by Flux) for process-parallel
# data loading over a `Dataset`.
#
# `DataLoader(ds; num_workers=N)` ships its data container to worker processes by serializing it
# from a background feeder task that does **not** hold the Python GIL; a `Dataset` serializes by
# calling `pickle`/PythonCall, which segfaults off the GIL-holding main task. The fix is to
# precompute the `pickle` bytes up front, on the main task, and ship *those* — pure Julia, safe
# from any thread. `DistributedDataset` is that prepared wrapper, and the `DataLoader(::Dataset)`
# method below installs it automatically when `num_workers > 0`.

"""
    DistributedDataset(ds::Dataset)

A [`Dataset`](@ref) prepared to be sent to `Distributed` worker processes. Construction
precomputes — on the current, GIL-holding task — the `pickle` bytes that carry `ds` (its data,
by reference to the on-disk Arrow files, plus its Python format). Serializing the wrapper later
just ships those bytes and touches no Python, so it is safe from the GIL-less feeder task of a
process-parallel data loader.

You rarely construct one directly: `MLUtils.DataLoader(ds; num_workers=N)` wraps `ds` in a
`DistributedDataset` for you. It forwards the `numobs`/`getobs` data-container interface to the
underlying dataset, so it is a drop-in stand-in for `ds`.
"""
struct DistributedDataset
    ds::Dataset             # valid on the process that built it; used for numobs/getobs there
    bytes::Vector{UInt8}    # precomputed pickle payload (data by reference + Python format)
    jltransform             # rides alongside the bytes
end

DistributedDataset(ds::Dataset) =
    DistributedDataset(ds, _pickle_bytes(ds), getfield(ds, :jltransform))

# Forward the data-container interface to the wrapped dataset (on the main process at
# construction, on the worker after `deserialize` has rebuilt a live `Dataset`).
MLUtils.numobs(dd::DistributedDataset) = numobs(dd.ds)
MLUtils.getobs(dd::DistributedDataset, i) = getobs(dd.ds, i)

# Ship the precomputed bytes — pure Julia, safe from the GIL-less feeder thread — never a `Py`.
function Serialization.serialize(s::AbstractSerializer, dd::DistributedDataset)
    Serialization.serialize_type(s, DistributedDataset)
    Serialization.serialize(s, dd.bytes)
    Serialization.serialize(s, dd.jltransform)
    return nothing
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{DistributedDataset})
    bytes       = Serialization.deserialize(s)
    jltransform = Serialization.deserialize(s)
    ds = set_jltransform!(Dataset(_unpickle_py(bytes)), jltransform)   # worker holds the GIL here
    return DistributedDataset(ds, bytes, jltransform)
end

# Wrap in `DistributedDataset` for process-parallel loading so the feeder-thread `serialize` ships
# precomputed bytes instead of calling Python. Serial and thread-parallel (`parallel`) loaders
# serialize nothing on the main process, so they get the raw `ds`. `invoke` reaches the generic
# `DataLoader(::Any; …)` constructor (the wrapped data is not a `Dataset`, so no recursion).
function MLUtils.DataLoader(ds::Dataset; num_workers::Integer = 0, kws...)
    data = num_workers > 0 ? DistributedDataset(ds) : ds
    return invoke(MLUtils.DataLoader, Tuple{Any}, data; num_workers, kws...)
end
