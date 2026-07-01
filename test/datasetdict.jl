
mnist = load_dataset("ylecun/mnist")

@test mnist isa DatasetDict
@test length(mnist) == 2

@testset "dictionary interface" begin
    @test DatasetDict <: AbstractDict{String, Dataset}
    @test eltype(mnist) == Pair{String, Dataset}
    @test keytype(mnist) == String
    @test valtype(mnist) == Dataset
    @test Set(keys(mnist)) == Set(["train", "test"])
    @test keys(mnist) isa Vector{String}
    @test all(v -> v isa Dataset, values(mnist))
    @test length(values(mnist)) == 2
    @test Set(first.(pairs(mnist))) == Set(["train", "test"])
    @test all(p -> p.second isa Dataset, pairs(mnist))
    @test haskey(mnist, "train")
    @test !haskey(mnist, "nonexistent")
    @test get(mnist, "train", nothing) isa Dataset
    @test get(mnist, "nonexistent", nothing) === nothing
    @test Set(k for (k, v) in mnist) == Set(["train", "test"])
end

@testset "display mirrors python repr" begin
    # `text/plain` show should match the Python `datasets.DatasetDict` repr, not the
    # generic `AbstractDict` multi-line display.
    s = sprint(show, MIME("text/plain"), mnist)
    @test startswith(s, "DatasetDict({")
    @test occursin("train: Dataset({", s)
    @test s == sprint(show, mnist)   # same as the 2-arg show
end

@testset "raw Python observations via set_format!(d, nothing)" begin
    @test_throws MethodError mnist[1]
    @test mnist["test"] isa Dataset
    ds = set_format!(copy(mnist), nothing)["test"]   # strip formatting -> raw python
    @test pyisinstance(ds[1], pytype(pydict()))
    @test ds[1]["image"] isa Py
    @test ds[1]["label"] isa Py
    @test pyisinstance(ds[1]["label"], pytype(pyint()))
    @test py2jl(ds[1]["label"]) == 7
    @test py2jl(ds[2]["label"]) == 2
    @test mnist["test"].format["type"] == "numpy"   # original untouched
end

@testset "with_format(julia)" begin
    d = with_format(mnist, "julia")
    ds = d["test"]
    @test ds.format["type"] == "numpy"
    x = ds[1]
    @test x isa Dict
    @test x["label"] isa Int
    @test x["label"] == 7
    @test x["image"] isa AbstractMatrix{UInt8}   # numpy format: raw array, not a colorview
    @test size(x["image"]) == (28, 28)
end

@testset "python transforms" begin
    @pyexec """
    def pytr(x):
        return {"label": [-l for l in x["label"]]}
    """ => pytr
    d = mnist.with_transform(pytr)
    @test d isa DatasetDict
    @test Bool(d["test"][1]["label"] == -7)
end

@testset "getproperty returns julia types" begin
    @test mnist.num_rows isa Dict{String, Int}
    @test mnist.num_rows == Dict("test"  => 10000, "train" => 60000)
end

@testset "set_format" begin
    d = deepcopy(mnist)
    set_format!(d, nothing)
    @test d["test"].format["type"] === nothing              # the copy is stripped ...
    @test mnist["test"].format["type"] == "numpy"           # ... original julia intact
    d.set_format("numpy")
    @test d["test"].format["type"] == "numpy"
    set_format!(d, nothing)
    @test d["test"].format["type"] === nothing
end

@testset "reset_format! / set_format! (DatasetDict)" begin
    d = set_format!(copy(mnist), nothing)                   # raw python
    @test pyisinstance(d["test"][1], pytype(pydict()))
    @test d["test"].format["type"] === nothing
    reset_format!(d)                                        # reset -> default julia
    @test d["test"][1] isa Dict
    @test d["test"].format["type"] == "numpy"
    set_format!(d, nothing)                                 # explicit nothing -> raw
    @test d["test"].format["type"] === nothing
    set_format!(d)                                          # single-arg form -> julia
    @test d["test"].format["type"] == "numpy"
    @test mnist["test"].format["type"] == "numpy"           # original untouched
end

@testset "merge / filter preserve the wrapper type" begin
    # `merge`/`filter` must return a `DatasetDict`, not a plain `Dict{String,Dataset}`.
    f = filter(p -> p.first == "train", mnist)
    @test f isa DatasetDict
    @test collect(keys(f)) == ["train"]

    extra = load_dataset("ylecun/mnist")["test"]   # a lone `Dataset`
    m = merge(mnist, Dict("extra" => extra))
    @test m isa DatasetDict
    @test Set(keys(m)) == Set(["train", "test", "extra"])

    # later dicts win, matching `Base.merge` semantics
    m2 = merge(mnist, Dict("train" => extra))
    @test m2 isa DatasetDict
    @test length(m2["train"]) == length(extra)

    # the `jltransform` of the first argument is carried over
    dj = with_format(mnist, "julia")
    fj = filter(p -> p.first == "test", dj)
    @test fj isa DatasetDict
    @test fj["test"][1] isa Dict
    @test fj["test"][1]["label"] isa Int
end

@testset "set_jltransform!" begin
    d = deepcopy(mnist)
    set_jltransform!(d) do x
        x = py2jl(x)
        x["label"] .= 1
        return x
    end

    @test d["test"][1:2]["label"] == [1, 1]
end
