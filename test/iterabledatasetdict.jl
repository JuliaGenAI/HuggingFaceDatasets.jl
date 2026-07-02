# Offline: an `IterableDatasetDict` is assembled from per-split iterables, mirroring what
# `load_dataset(...; streaming=true)` returns when no split is selected.

@testset "IterableDatasetDict" begin
    tr = Dataset((; x = [1, 2, 3])).py.to_iterable_dataset()
    te = Dataset((; x = [4, 5])).py.to_iterable_dataset()
    pyidd = datasets.IterableDatasetDict(pydict(Dict("train" => tr, "test" => te)))
    idd = set_format!(IterableDatasetDict(pyidd), "julia")   # as load_dataset does

    @test idd isa IterableDatasetDict
    @test IterableDatasetDict <: AbstractDict{String, IterableDataset}
    @test Set(keys(idd)) == Set(["train", "test"])
    @test keys(idd) isa Vector{String}
    @test length(idd) == 2
    @test haskey(idd, "train")
    @test !haskey(idd, "nope")
    @test get(idd, "nope", 42) == 42
    @test get(idd, "train", nothing) isa IterableDataset

    train = idd["train"]
    @test train isa IterableDataset
    @test collect(train) == [Dict("x" => 1), Dict("x" => 2), Dict("x" => 3)]

    # iteration yields split-name => IterableDataset pairs
    @test all(p -> p.second isa IterableDataset, pairs(idd))
    @test Set(k for (k, v) in idd) == Set(["train", "test"])

    # map over every split, bridging Julia values
    idd2 = idd.map(row -> Dict("x" => row["x"] + 10))
    @test idd2 isa IterableDatasetDict
    @test collect(idd2["test"]) == [Dict("x" => 14), Dict("x" => 15)]

    # filter examples within every split
    idd3 = idd.filter(row -> row["x"] > 2)
    @test collect(idd3["train"]) == [Dict("x" => 3)]
    @test collect(idd3["test"]) == [Dict("x" => 4), Dict("x" => 5)]
end
