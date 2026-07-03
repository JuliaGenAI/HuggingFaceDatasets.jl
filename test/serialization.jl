using Serialization: serialize, deserialize
using Distributed
using MLUtils: DataLoader, getobs, numobs
import MLUtils

# serialize → deserialize through an in-memory buffer (the same path `Distributed` uses to
# ship a `Dataset` to a worker, minus the process hop). A worker read is exercised
# separately by the cross-process getobs test.
roundtrip(ds) = (io = IOBuffer(); serialize(io, ds); seekstart(io); deserialize(io))

@testset "julia format round-trip" begin
    ds = Dataset((; x = reshape(collect(1:8*20), 8, 20), label = collect(0:19)))
    ds2 = roundtrip(ds)
    @test ds2 isa Dataset
    @test length(ds2) == length(ds) == 20
    @test ds2[1:20]["label"] == ds[1:20]["label"]
    @test ds2[1:20]["x"] == ds[1:20]["x"]
    @test ds2[3]["x"] isa Vector{Int}          # "julia" format preserved → native Julia arrays
    @test ds2[3] == ds[3]
end

@testset "format preserved across round-trip" begin
    ds = Dataset((; label = collect(0:4)))     # julia format
    @test roundtrip(ds)[1]["label"] isa Int    # julia → native scalar

    dsn = set_format!(copy(ds), nothing)       # raw Python
    r = roundtrip(dsn)
    @test r[1] isa Py                          # nothing-format preserved (raw Python back)
    @test pyconvert(Int, r[1]["label"]) == 0

    dsnp = set_format!(copy(ds), "numpy")      # plain numpy (identity transform)
    @test roundtrip(dsnp)[1] isa Py
end

@testset "data saved once per fingerprint" begin
    ds = Dataset((; x = collect(1:10)))
    n0 = length(HuggingFaceDatasets._SAVE_CACHE)
    roundtrip(ds); roundtrip(ds)               # same content ⇒ a single on-disk copy
    @test length(HuggingFaceDatasets._SAVE_CACHE) == n0 + 1
end

@testset "DistributedDataset round-trip" begin
    # The feeder-safe wrapper installed by `DataLoader(ds; num_workers>0)`: it precomputes the
    # pickle bytes on this task and serializes *those* (no `Py`, no Python call), so it survives
    # being shipped from a GIL-less thread. Round-tripping must rebuild a working dataset.
    ds = Dataset((; x = reshape(collect(1:8*20), 8, 20), label = collect(0:19)))
    dd = DistributedDataset(ds)
    @test dd isa DistributedDataset
    @test numobs(dd) == 20
    r = roundtrip(dd)
    @test r isa DistributedDataset
    @test getobs(r, 1:20)["label"] == ds[1:20]["label"]
    @test getobs(r, 1:20)["x"] == ds[1:20]["x"]
end

# The real cross-process paths (serialize → ship to a worker → re-mmap → read there) each spin
# up a worker process with its own Python. They guard invariants the in-process round-trips
# above cannot: that no `Py` crosses the boundary (a stray `Py` round-trips fine within one
# interpreter but segfaults across one), and that materializing an in-memory dataset does not
# crash a concurrent loader. Worth the few extra seconds on CI.
@testset "cross-process round-trip" begin
    procs = addprocs(1; exeflags = "--project=$(dirname(Base.active_project()))")
    try
        @everywhere procs using HuggingFaceDatasets
        ds = Dataset((; x = reshape(collect(1:8*20), 8, 20), label = collect(0:19)))
        got = remotecall_fetch(d -> d[1:20], only(procs), ds)   # ds serialized to the worker
        @test got["x"] == ds[1:20]["x"]
        @test got["label"] == ds[1:20]["label"]
    finally
        rmprocs(procs)
    end
end

# End-to-end process-parallel loading of an *in-memory* dataset via MLUtils. This drives the
# full `DataLoader(...; num_workers=N)` path: the `DataLoader(::Dataset)` hook wraps `ds` in a
# `DistributedDataset` (precomputing the pickle bytes on the main task) so the background feeder
# ships those bytes instead of calling Python off the GIL-holding thread, which used to segfault.
@testset "num_workers DataLoader over in-memory dataset" begin
    ds = Dataset((; x = reshape(collect(1:8*40), 8, 40), label = collect(0:39)))
    loader = DataLoader(ds; batchsize = 10, num_workers = 2)
    try
        batches = collect(loader)                      # iterates ⇒ ships ds to 2 workers
        @test length(batches) == 4
        @test sort(reduce(vcat, [b["label"] for b in batches])) == collect(0:39)
    finally
        MLUtils.close_dataloader_pool()
    end
end
