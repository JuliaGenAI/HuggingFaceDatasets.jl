using Serialization: serialize, deserialize
using Distributed

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

# The actual cross-process path (serialize → ship to a worker → re-mmap → read there) needs
# an extra worker process, each spinning up its own Python — too heavy for CI. It guards the
# invariant that no `Py` crosses the boundary, which the in-process round-trips above cannot
# catch (a stray `Py` round-trips fine within a single interpreter but segfaults across one).
if !parse(Bool, get(ENV, "CI", "false"))
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
end
