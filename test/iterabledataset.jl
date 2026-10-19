# All tests here stay offline: an `IterableDataset` is built from in-memory data via
# `Dataset(...).to_iterable_dataset()`, mirroring what `load_dataset(...; streaming=true)`
# returns for a single split.

@testset "IterableDataset basics" begin
    ds = Dataset((; x = [1, 2, 3], y = [10, 20, 30]))
    itds = ds.to_iterable_dataset()          # forwarded method → wrapped IterableDataset

    @test itds isa IterableDataset
    @test Base.IteratorSize(IterableDataset) == Base.SizeUnknown()

    # the "julia" format is re-attached by the forwarded call, so rows are native Julia dicts
    @test collect(itds) == [Dict("x" => 1, "y" => 10),
                            Dict("x" => 2, "y" => 20),
                            Dict("x" => 3, "y" => 30)]

    # iteration and lazy materialization
    @test collect(Iterators.take(itds, 2)) == [Dict("x" => 1, "y" => 10),
                                               Dict("x" => 2, "y" => 20)]
    acc = Dict{String,Int}[]
    for obs in itds
        push!(acc, obs)
    end
    @test length(acc) == 3

    # no random access / no length: explanatory ArgumentError, not a bare MethodError
    @test_throws ArgumentError length(itds)
    @test_throws ArgumentError itds[1]
    @test_throws ArgumentError firstindex(itds)
    @test_throws ArgumentError lastindex(itds)
end

@testset "lazy forwarded methods re-wrap" begin
    ds = Dataset((; x = [1, 2, 3, 4, 5]))
    itds = ds.to_iterable_dataset()

    @test itds.take(2) isa IterableDataset
    @test collect(itds.take(2)) == [Dict("x" => 1), Dict("x" => 2)]

    @test itds.skip(3) isa IterableDataset
    @test collect(itds.skip(3)) == [Dict("x" => 4), Dict("x" => 5)]

    sh = itds.shuffle(seed = 0, buffer_size = 5)
    @test sh isa IterableDataset
    @test sort([r["x"] for r in sh]) == [1, 2, 3, 4, 5]   # a permutation of the same rows
end

@testset "map / filter (julia-bridged)" begin
    ds = Dataset((; x = [1, 2, 3, 4, 5]))
    itds = ds.to_iterable_dataset()

    # property-style and function-form both bridge Julia values and stay lazy
    m = itds.map(row -> Dict("x" => row["x"] + 100))
    @test m isa IterableDataset
    @test collect(m) == [Dict("x" => v) for v in 101:105]   # scalars, not 0-d arrays

    m2 = map(row -> Dict("x" => row["x"] * 2), itds)
    @test collect(m2) == [Dict("x" => v) for v in [2, 4, 6, 8, 10]]

    f = itds.filter(row -> row["x"] > 2)
    @test f isa IterableDataset
    @test [r["x"] for r in f] == [3, 4, 5]

    f2 = filter(row -> row["x"] == 5, itds)
    @test collect(f2) == [Dict("x" => 5)]
end

@testset "julia format keeps each column's dtype" begin
    x = rand(3, 4)
    itds = Dataset((; x, f = [1.0000000001, 2.0, 3.0, 4.0])).to_iterable_dataset()
    rows = collect(itds)
    @test rows[1]["x"] isa Vector{Float64} && rows[1]["x"] == x[:, 1]
    @test rows[1]["f"] === 1.0000000001
end

@testset "julia format reads string columns as Julia strings" begin
    itds = Dataset((; s = ["a", "b"^1000, "c"])).to_iterable_dataset()
    @test [r["s"] for r in itds] == ["a", "b"^1000, "c"]
    @test first(itds)["s"] isa String
end

@testset "format switching" begin
    ds = Dataset((; x = [1, 2, 3]))
    itds = ds.to_iterable_dataset()

    # opt out to raw Python observations
    raw = set_format!(copy(itds), nothing)
    @test first(raw) isa Py
    @test itds !== raw
    # the original is untouched (copy-on-write via `copy`)
    @test first(itds) == Dict("x" => 1)
    @test first(with_format(itds, nothing)) isa Py
    @test first(itds) == Dict("x" => 1)

    # opt back in
    j = with_format(raw, "julia")
    @test first(j) == Dict("x" => 1)

    # property-style set_format routes to the julia method (would error if forwarded to Python)
    itds.set_format("julia")
    @test first(itds) == Dict("x" => 1)

    # custom jltransform via with_jltransform
    tr = with_jltransform(itds, row -> py2jl(row)["x"] + 1)
    @test collect(tr) == [2, 3, 4]
end
