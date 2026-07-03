# Schema views (`Features`/`ClassLabel`/`Value`) and the `class_names`/`int2str`/`str2int`
# helpers. All local (no network), so these run in CI. `Features`/`ClassLabel` are exported
# (via `runtests.jl`'s `using`); the rest are public but unexported, so bring them into scope.
using HuggingFaceDatasets: Value, features, class_names, int2str, str2int

# A tiny classification dataset: `class_encode_column` turns the string column into a
# `ClassLabel` (names sorted: "cat" => 0, "dog" => 1).
_encoded() = Dataset((; label = ["cat", "dog", "dog", "cat"], x = [10, 20, 30, 40])).class_encode_column("label")

@testset "Features view" begin
    ds = _encoded()

    f = features(ds)
    @test f isa Features
    @test f isa AbstractDict
    @test ds.features isa Features            # `ds.features` getproperty branch, not raw Py

    @test Set(keys(f)) == Set(["label", "x"])
    @test length(f) == 2
    @test haskey(f, "label")
    @test haskey(f, :label)                   # Symbol key
    @test !haskey(f, "missing")

    # iteration yields name => feature pairs (so `Dict(f)`, `collect`, ... work)
    d = Dict(f)
    @test Set(keys(d)) == Set(["label", "x"])
    @test d["label"] isa ClassLabel
    @test d["x"] isa Value

    # the rest of the `AbstractDict` surface the docstring promises
    @test values(f) |> collect |> length == 2
    @test get(f, "label", nothing) isa ClassLabel
    @test get(f, "missing", :deflt) === :deflt
    @test_throws KeyError f["missing"]        # not a raw Python `PyException`

    # keys/length are Python-free (answered from the cached names) — safe for tab-completion
    @test keys(f) == ["label", "x"]
end

@testset "ClassLabel view" begin
    ds = _encoded()
    cl = ds.features["label"]
    @test cl isa ClassLabel

    # attribute/method access forwards to Python (the primary, Pythonic idiom), converted by
    # py2jl — including vector args
    @test cl.names == ["cat", "dog"]
    @test cl.num_classes == 2
    @test cl.int2str(1) == "dog"
    @test cl.int2str([0, 1, 1]) == ["cat", "dog", "dog"]
    @test cl.str2int("cat") == 0

    # Julian convenience functions on (ds, col)
    @test class_names(ds, "label") == ["cat", "dog"]
    @test class_names(ds, :label) == ["cat", "dog"]   # Symbol column

    # 0-based class ids pass through with NO offset (labels are data, not Julia indices)
    @test int2str(ds, "label", 0) == "cat"
    @test int2str(ds, "label", 1) == "dog"
    @test str2int(ds, "label", "dog") == 1

    # vector arguments decode/encode a whole batch in one call
    @test int2str(ds, "label", [0, 1, 1]) == ["cat", "dog", "dog"]
    @test str2int(ds, "label", ["cat", "dog"]) == [0, 1]

    # decoding a whole column matches `names[labels .+ 1]` (the 0-based id -> 1-based bridge)
    labels = collect(ds["label"])
    @test int2str(ds, "label", labels) == class_names(ds, "label")[labels .+ 1]
end

@testset "Value view" begin
    ds = Dataset((; x = [1, 2, 3], f = [1.0, 2.0, 3.0], s = ["a", "b", "c"]))
    @test ds.features["x"] isa Value
    @test ds.features["x"].dtype == "int64"
    @test ds.features["f"].dtype == "float64"
    @test ds.features["s"].dtype == "string"
end

@testset "non-ClassLabel column errors clearly" begin
    ds = Dataset((; x = [1, 2, 3]))
    @test_throws ArgumentError class_names(ds, "x")
    @test_throws ArgumentError int2str(ds, "x", 0)
    @test_throws ArgumentError str2int(ds, "x", "a")
end

@testset "construct schema from Julia + round-trip through features=" begin
    cl = ClassLabel(names = ["neg", "pos"])
    @test cl isa ClassLabel
    @test cl.names == ["neg", "pos"]
    @test cl.num_classes == 2

    v = Value("int64")
    @test v isa Value
    @test v.dtype == "int64"

    sch = Features(Dict("label" => cl, "x" => v))
    @test sch isa Features
    @test sch["label"] isa ClassLabel
    @test sch["x"] isa Value

    # jl2py unwraps the views back to the underlying Python objects
    @test pyis(jl2py(cl), cl.py)
    @test pyis(jl2py(sch), sch.py)

    # feed the Julia-built schema back into construction via `features=`
    ds = Dataset.from_dict(Dict("label" => [0, 1, 1, 0], "x" => [1, 2, 3, 4]);
                           features = jl2py(sch))
    @test ds.features["label"] isa ClassLabel
    @test class_names(ds, "label") == ["neg", "pos"]
    @test int2str(ds, "label", 0) == "neg"
    @test int2str(ds, "label", 1) == "pos"
end
