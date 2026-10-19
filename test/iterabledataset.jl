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

    # batched filter with a `BitVector` mask (handed to Python as a numpy bool array)
    fb = itds.filter(b -> isodd.(1:length(b["x"])); batched=true)
    @test [r["x"] for r in fb] == [1, 3, 5]

    # numeric vectors returned by `f` keep their element type (#80) and are copied, so a
    # yielded row doesn't alias the vector `f` returned
    v = Float32[1, 2]
    mv = itds.map(row -> Dict("y" => v))
    r = first(mv.py.with_format(nothing))
    @test pyconvert(String, r["y"].dtype.name) == "float32"
    @test !pyconvert(Bool, np.shares_memory(r["y"], jl2numpy(v)))
    y = first(mv)["y"]
    @test y == [1, 2]
    y[1] = 100
    @test v == [1, 2]

    # ... and so do narrow numeric scalars (#93)
    ms = itds.map(row -> Dict("u" => UInt16(row["x"]), "f" => Float32(row["x"])))
    r = first(ms.py.with_format(nothing))
    @test pyconvert(String, r["u"].dtype.name) == "uint16"
    @test pyconvert(String, r["f"].dtype.name) == "float32"
    @test [r["u"] for r in ms] == 1:5

    # N-D arrays are copied too: rows built in a reused buffer stay distinct once collected (#94)
    buf = zeros(Float32, 2, 2)
    rows = collect(itds.map(row -> (buf .= row["x"]; Dict("m" => buf))))
    @test [r["m"] for r in rows] == [fill(x, 2, 2) for x in 1:5]
    rows[1]["m"][1] = 100
    @test buf == fill(5, 2, 2)
end

@testset "map / filter callbacks see py2jl values (#91)" begin
    itds = Dataset((; x = UInt16[1, 2, 3, 4])).to_iterable_dataset()
    seen = Ref{Any}()

    # a batched map re-batches formatted rows: the column is a typed `Array`, not an `object`
    # array of numpy scalars (as with `datasets` < 5.1)
    collect(itds.map(b -> (seen[] = b; b); batched=true, batch_size=2))
    @test seen[]["x"] isa Vector{UInt16} && seen[]["x"] == [3, 4]
    m = itds.map(b -> Dict("y" => UInt8.(b["x"])); batched=true, batch_size=2)
    @test [r["y"] for r in m] == [1, 2, 3, 4]

    collect(itds.map(x -> (seen[] = x; x)))
    @test seen[] == Dict("x" => UInt16(4))
    @test first(itds)["x"] isa UInt16                   # rows keep the column's type too

    @test [r["x"] for r in itds.filter((x, i) -> iseven(i); with_indices=true)] == [1, 3]

    # the `"julia"` formatter stacks a list of numpy scalars by dtype
    fmt = datasets.formatting.get_formatter("julia")
    a = fmt._consolidate(pylist([np.uint16(1), np.uint16(2)]))
    @test pyconvert(String, a.dtype.name) == "uint16"
    @test pyconvert(String, fmt._consolidate(pylist([np.uint16(1), np.int8(2)])).dtype.name) == "object"
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

@testset "batch iterator (#105, #107)" begin
    ds = Dataset((; text = ["a", "bb", "ccc"], x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    itds = ds.to_iterable_dataset()

    it = itds.iter(batch_size = 2)
    @test Base.IteratorSize(it) == Base.SizeUnknown()
    batches = collect(it)
    @test batches == [ds[1:2], ds[3:3]]
    @test batches[1]["text"] isa Vector{String} && batches[1]["x"] isa Matrix{Float64}
    @test collect(itds.iter(batch_size = 2, drop_last_batch = true)) == batches[1:1]
    @test [b["text"] for b in itds.take(2).iter(batch_size = 5)] == [["a", "bb"]]

    raw = with_format(itds, nothing)
    texts = [b["text"] for b in raw.iter(batch_size = 2)]
    @test all(t -> pyis(t, pybuiltins.None), texts)   # released as the loop advanced
    @test [pyconvert(Vector{String}, copy(b)["text"]) for b in raw.iter(batch_size = 2)] ==
          [["a", "bb"], ["ccc"]]
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
