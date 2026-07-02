
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

@testset "construction from Julia data" begin
    train = Dataset((; label=[1, 0, 1, 0]))
    test  = Dataset((; label=[1, 1]))

    dd = DatasetDict("train" => train, "test" => test)      # Pair varargs
    @test dd isa DatasetDict
    @test Set(keys(dd)) == Set(["train", "test"])
    @test dd["train"] isa Dataset
    @test length(dd["train"]) == 4

    dd2 = DatasetDict(Dict("train" => train, "test" => test))  # AbstractDict
    @test Set(keys(dd2)) == Set(["train", "test"])

    # each split keeps its source `Dataset`'s transform: `Dataset((;...))` is julia by
    # default, so observations come back as native Julia values
    @test dd["train"][1] == Dict("label" => 1)
    @test dd["train"].format["type"] == "numpy"

    # per split: a julia source stays julia, a raw source stays raw
    raw = set_format!(Dataset((; label=[5, 0, 4])), nothing)
    @test raw.format["type"] === nothing
    mixed = DatasetDict("j" => train, "r" => raw)
    @test mixed["j"][1] == Dict("label" => 1)   # julia split -> Julia Dict
    @test mixed["r"][1] isa Py                   # raw split   -> raw Python

    # copy-on-write: changing the dict's format must not mutate the source `Dataset`
    built = DatasetDict("t" => train)
    set_format!(built, nothing)
    @test built["t"][1] isa Py
    @test train.format["type"] == "numpy"        # source untouched

    # the public constructors take no `jltransform` kwarg (it is derived from the splits)
    @test_throws MethodError DatasetDict("train" => train; jltransform = identity)
end

@testset "per-split jltransform" begin
    # range indexing returns the transform's output directly, so a constant-returning
    # transform is an unambiguous marker of which transform ran on which split
    dd = DatasetDict("a" => Dataset((; x=[1, 2, 3])), "b" => Dataset((; x=[4, 5])))

    # a Dict sets a different transform per split; omitted splits fall back to identity
    set_jltransform!(dd, Dict("a" => (o -> "TA")))
    @test dd["a"][1:3] == "TA"                     # split "a" uses its transform
    @test dd["b"][1:2] isa Py                      # "b" omitted -> identity -> raw batch

    # with_jltransform accepts a per-split dict and does not mutate the original
    dd4 = with_jltransform(dd, Dict("a" => (o -> "X"), "b" => (o -> "Y")))
    @test dd4["a"][1:3] == "X"
    @test dd4["b"][1:2] == "Y"
    @test dd["a"][1:3] == "TA"                     # original unchanged

    # a single callable still broadcasts to every split
    set_jltransform!(dd, o -> "SAME")
    @test dd["a"][1:3] == "SAME"
    @test dd["b"][1:2] == "SAME"
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

@testset "property-style format methods (python interface)" begin
    # `d.set_format`/`d.reset_format`/`d.with_format` route to this package's methods, so the
    # `"julia"` pseudo-format works across all splits through the Python-style calls.
    d = set_format!(copy(mnist), nothing)
    @test d["test"].format["type"] === nothing
    d.set_format("julia")                                   # would error if forwarded to Python
    @test d["test"].format["type"] == "numpy"
    @test d["test"][1] isa Dict
    set_format!(d, nothing)
    d.reset_format()
    @test d["test"].format["type"] == "numpy"
    @test d.with_format("julia") isa DatasetDict
end

@testset "merge preserves the wrapper type" begin
    # `merge` must return a `DatasetDict`, not a plain `Dict{String,Dataset}`.
    extra = load_dataset("ylecun/mnist")["test"]   # a lone `Dataset`
    m = merge(mnist, Dict("extra" => extra))
    @test m isa DatasetDict
    @test Set(keys(m)) == Set(["train", "test", "extra"])

    # later dicts win, matching `Base.merge` semantics
    m2 = merge(mnist, Dict("train" => extra))
    @test m2 isa DatasetDict
    @test length(m2["train"]) == length(extra)
end

@testset "julia-friendly map / filter (DatasetDict)" begin
    dd = DatasetDict("train" => Dataset((; label=[5, 0, 4, 3])),
                     "test"  => Dataset((; label=[1, 2])))

    # `map(f, dd)` and `dd.map(f)` both bridge Julia values, per-example over every split
    dm = map(x -> Dict("label" => x["label"] + 10), dd)
    @test dm isa DatasetDict
    @test Set(keys(dm)) == Set(["train", "test"])
    @test dm["train"][1:4]["label"] == [15, 10, 14, 13]
    @test dm["test"][1:2]["label"] == [11, 12]
    @test dd.map(x -> Dict("label" => x["label"] + 10))["train"][1:4]["label"] == [15, 10, 14, 13]

    # `filter(f, dd)` and `dd.filter(f)` both filter EXAMPLES within every split (python
    # `DatasetDict.filter`) — the function form no longer filters splits.
    df = filter(x -> x["label"] > 2, dd)
    @test df isa DatasetDict
    @test Set(keys(df)) == Set(["train", "test"])       # both splits kept
    @test sort(df["train"][1:length(df["train"])]["label"]) == [3, 4, 5]
    @test length(df["test"]) == 0                        # none > 2
    # the property form is equivalent
    dfp = dd.filter(x -> x["label"] > 2)
    @test sort(dfp["train"][1:length(dfp["train"])]["label"]) == [3, 4, 5]
    @test length(dfp["test"]) == 0

    # batched map keyword is forwarded
    db = dd.map(x -> Dict("label" => x["label"] .* 2); batched=true)
    @test db["train"][1:4]["label"] == [10, 0, 8, 6]
end

@testset "keys/length are cached and stay correct" begin
    # `keys`/`length`/`haskey` are served from the cached split names (no Python call), which
    # is what makes REPL `d[<TAB>` completion safe. Guard that the cache is correct and stays
    # in sync with the underlying python object across (re)construction.
    dd = DatasetDict("a" => Dataset((; x=[1, 2])), "b" => Dataset((; x=[3])))
    @test keys(dd) == ["a", "b"]                 # cached, ordered
    @test length(dd) == 2
    @test haskey(dd, "a") && !haskey(dd, "z")
    @test keys(dd) == [pyconvert(String, k) for k in dd.py.keys()]   # matches python truth

    # reconstruction via a forwarded/bridged method keeps the cache correct
    dm = map(x -> Dict("x" => x["x"] .+ 1), dd)
    @test keys(dm) == ["a", "b"] && length(dm) == 2
    @test keys(deepcopy(dd)) == ["a", "b"]
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
