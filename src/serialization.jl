# Cross-process / on-disk serialization for `Dataset`.
#
# A `Dataset` wraps a `Py`, which must never be serialized across a process boundary:
# PythonCall's generic `Py` serialization is unsafe here and can segfault a worker. Instead
# we use `datasets`' **own pickle** — the exact mechanism the Python library relies on for
# `torch.utils.data.DataLoader(num_workers>0)`. An on-disk (e.g. Hub) dataset pickles to a
# ~1 KB *reference* to its memory-mapped Arrow `cache_files` and re-mmaps on the other side,
# so the data is shared, not copied. Only the pickle bytes and the Julia transform cross
# Julia's `Distributed` boundary — never a `Py`.
#
# This makes `Dataset` a first-class data container for process-parallel data loaders (e.g.
# a `MLUtils.DataLoader(ds; num_workers=N)` that spreads `getobs` over worker processes):
# the loader serializes `ds` to each worker, where it is reconstructed and read with that
# worker's own Python interpreter — and independent GILs let those reads run in parallel,
# which threads cannot under the shared GIL.
#
# Requirements/limitations:
# - The dataset must be read in a Julia-native format (the default `"julia"`) so that
#   `getobs` returns serializable Julia arrays; a raw-`Py` format is not serializable back.
# - A custom `jltransform` must itself be serializable (a plain/named function is; an
#   anonymous closure capturing a `Py` is not).
# - Workers re-mmap the referenced files, so they must share a filesystem with the main
#   process (same node) — the same constraint as PyTorch's fork/spawn workers.

import Serialization
using Serialization: AbstractSerializer

# Temp Arrow directories backing *in-memory* datasets (those built from Julia data, with no
# `cache_files`), keyed by content fingerprint so each is written once no matter how many
# workers/loaders request it — then it, too, pickles by reference. Removed at process exit
# (`mktempdir`).
const _SAVE_CACHE = Dict{String,String}()

# A `datasets.Dataset` `Py` backed by on-disk Arrow files, so `pickle` references the files
# (zero-copy re-mmap on the worker) instead of embedding the in-memory buffer. On-disk
# datasets (Hub / `load_from_disk` / `from_*`) already qualify; in-memory ones are
# materialized once.
function _ondisk_py(ds::Dataset)
    py = getfield(ds, :py)
    isempty(pyconvert(Vector, py.cache_files)) || return py
    dir = get!(_SAVE_CACHE, pyconvert(String, py._fingerprint)) do
        d = mktempdir(); py.save_to_disk(d); d          # auto-removed at process exit
    end
    return getfield(load_from_disk(dir)::Dataset, :py)
end

function Serialization.serialize(s::AbstractSerializer, ds::Dataset)
    Serialization.serialize_type(s, Dataset)
    # `pickle` carries the data-by-reference AND the Python format; the Julia transform is
    # a separate field of the wrapper, so it rides alongside.
    Serialization.serialize(s, pyconvert(Vector{UInt8}, pickle.dumps(_ondisk_py(ds))))
    Serialization.serialize(s, getfield(ds, :jltransform))
    return nothing
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{Dataset})
    bytes       = Serialization.deserialize(s)
    jltransform = Serialization.deserialize(s)
    py = pickle.loads(pybytes(bytes))                   # re-mmaps the referenced Arrow files
    return set_jltransform!(Dataset(py), jltransform)   # Python format already restored by pickle
end
