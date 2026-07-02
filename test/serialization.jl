using Serialization: serialize, deserialize

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
