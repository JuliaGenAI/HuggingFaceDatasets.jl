mnist = load_dataset("ylecun/mnist", split="test")
glue_ax = load_dataset("nyu-mll/glue", "ax", split="test")

@testset "mnist" begin
    @test mnist isa Dataset
    @test length(mnist) == 10000
    @test mnist.num_rows isa Int
    @test mnist.num_rows == 10000
end

@testset "glue_ax" begin
    @test glue_ax isa Dataset
    @test length(glue_ax) == 1104
end

@testset "returned attributes are converted to julia" begin
    @test mnist.format isa Dict # returned attributes are converted to julia
    @test mnist.format["type"] == "numpy"   # loaded in the "julia" (numpy-backed) format
end

@testset "firstindex / lastindex / iteration" begin
    ds = Dataset(HuggingFaceDatasets.datasets.Dataset.from_dict(
        pydict(Dict("x" => pylist([10, 20, 30])))))
    @test firstindex(ds) == 1
    @test lastindex(ds) == 3
    @test pyconvert(Int, ds[end]["x"]) == 30
    @test pyconvert(Int, ds[begin]["x"]) == 10
    @test [pyconvert(Int, o["x"]) for o in ds] == [10, 20, 30]   # iterates observations
    @test length(collect(ds)) == 3
end

@testset "construct from Julia data" begin
    # Dict of scalar vectors: the "julia" format is applied by default (no with_format)
    ds = Dataset(Dict("label" => [5, 0, 4]))
    @test ds isa Dataset
    @test length(ds) == 3
    @test ds.format["type"] == "numpy"                     # julia format by default
    @test collect(keys(pyconvert(Dict, ds.py.features))) == ["label"]
    @test ds[1] == Dict("label" => 5)
    @test ds[1:3]["label"] == [5, 0, 4]

    # NamedTuple path preserves column names and order
    ds = Dataset((label = [5, 0, 4], text = ["a", "b", "c"]))
    @test length(ds) == 3
    @test ds.column_names == ["label", "text"]
    @test ds[1:3]["label"] == [5, 0, 4]
    @test ds[1:3]["text"] == ["a", "b", "c"]

    # mixed scalar types each round-trip
    ds = Dataset((i = [1, 2], s = ["x", "y"], f = [1.5, 2.5], b = [true, false]))
    @test ds[1:2]["i"] == [1, 2]
    @test ds[1:2]["s"] == ["x", "y"]
    @test ds[1:2]["f"] == [1.5, 2.5]
    @test ds[1:2]["b"] == [true, false]

    # a custom `jltransform` opts out of the default julia format (raw Python format kept)
    ds = Dataset(Dict("label" => [5, 0, 4]); jltransform = py2jl)
    @test ds.format["type"] === nothing
    @test ds[1] isa Dict
    @test ds[1]["label"] == 5

    # N-D array columns given as a vector-of-arrays (one array per observation)
    # round-trip: a single row reads back as the original array and a range stacks them
    # along the last axis (obs axis last, matching MLUtils / the numpy read path).
    m1 = [1 2 3; 4 5 6]; m2 = [7 8 9; 10 11 12]  # each 2×3
    ds = Dataset((; x = [m1, m2]))
    # inferred schema is a nested `List` (W2), not a fixed-shape `Array2D`
    @test pyconvert(String, pystr(ds.py.features["x"])) == "List(List(Value('int64')))"
    @test ds[1]["x"] == m1
    @test ds[2]["x"] == m2
    b = ds[1:2]["x"]
    @test b isa Array{Int, 3}
    @test size(b) == (2, 3, 2)
    @test b[:, :, 1] == m1
    @test b[:, :, 2] == m2

    # N-D array column given as a single stacked `(dims…, N)` array (last axis = obs):
    # perfectly symmetric with the read path, `Dataset((; x = batch))[:]["x"] == batch`.
    batch = reshape(collect(1:24), 2, 3, 4)  # 4 observations of 2×3
    ds = Dataset((; x = batch))
    @test ds[:]["x"] == batch
    @test ds[1]["x"] == batch[:, :, 1]

    # non-array columns are rejected
    @test_throws ArgumentError Dataset(Dict("x" => 5))
end

@testset "construct from a Tables.jl source" begin
    # A row table (vector of NamedTuples) is Tables.jl-compatible and lands on the generic
    # `Dataset(table)` method; it must match the equivalent column construction.
    rt = [(label = 5, text = "a"), (label = 0, text = "b"), (label = 4, text = "c")]
    ds = Dataset(rt)
    @test length(ds) == 3
    @test ds.column_names == ["label", "text"]
    @test ds[1:3]["label"] == [5, 0, 4]
    @test ds[1:3]["text"] == ["a", "b", "c"]
    @test ds[1:3]["label"] == Dataset((label = [5, 0, 4], text = ["a", "b", "c"]))[1:3]["label"]

    # a custom `jltransform` is forwarded through the table path
    ds = Dataset(rt; jltransform = py2jl)
    @test ds.format["type"] === nothing
    @test ds[1]["label"] == 5

    # a non-table scalar is rejected with a clear error
    @test_throws ArgumentError Dataset(5)
end

@testset "top-level combinators and file constructors" begin
    a = Dataset((; label = [1, 2]))
    b = Dataset((; label = [3, 4, 5]))

    @testset "concatenate_datasets" begin
        # varargs and vector forms are equivalent; result is in the julia format
        ds = concatenate_datasets(a, b)
        @test ds isa Dataset
        @test ds.format["type"] == "numpy"          # default julia format
        @test length(ds) == 5
        @test ds[:]["label"] == [1, 2, 3, 4, 5]
        @test concatenate_datasets([a, b])[:]["label"] == [1, 2, 3, 4, 5]

        # column-wise concatenation (same number of rows)
        c = Dataset((; x = [10, 20]))
        wide = concatenate_datasets(a, c; axis = 1)
        @test Set(wide.column_names) == Set(["label", "x"])
        @test wide[1] == Dict("label" => 1, "x" => 10)
    end

    @testset "interleave_datasets" begin
        x = Dataset((; label = [1, 1, 1]))
        y = Dataset((; label = [2, 2, 2]))
        ds = interleave_datasets(x, y)
        @test ds isa Dataset
        @test ds.format["type"] == "numpy"
        @test ds[:]["label"] == [1, 2, 1, 2, 1, 2]
        @test interleave_datasets([x, y])[:]["label"] == [1, 2, 1, 2, 1, 2]
    end

    @testset "save_to_disk / load_from_disk round-trip" begin
        mktempdir() do dir
            path = joinpath(dir, "ds")
            a.save_to_disk(path)                     # write side is a forwarded method
            loaded = load_from_disk(path)            # read side is the new top-level wrapper
            @test loaded isa Dataset
            @test loaded.format["type"] == "numpy"   # default julia format
            @test loaded[:]["label"] == [1, 2]
        end
    end

    @testset "from_csv / from_json / from_parquet" begin
        mktempdir() do dir
            src = Dataset((label = [5, 0, 4], text = ["a", "b", "c"]))

            csv = joinpath(dir, "d.csv")
            src.py.to_csv(csv)
            ds = HuggingFaceDatasets.from_csv(csv)
            @test ds isa Dataset
            @test ds.format["type"] == "numpy"
            @test ds[:]["label"] == [5, 0, 4]
            @test ds[:]["text"] == ["a", "b", "c"]

            json = joinpath(dir, "d.json")
            src.py.to_json(json)
            ds = HuggingFaceDatasets.from_json(json)
            @test ds[:]["label"] == [5, 0, 4]

            parquet = joinpath(dir, "d.parquet")
            src.py.to_parquet(parquet)
            ds = HuggingFaceDatasets.from_parquet(parquet)
            @test ds[:]["label"] == [5, 0, 4]
        end
    end

    @testset "jl2py unwraps the wrapper types" begin
        # the inbound half of the one-rewrap boundary
        @test pyis(jl2py(a), getfield(a, :py))
        dd = DatasetDict("train" => a)
        @test pyis(jl2py(dd), getfield(dd, :py))
        col = a["label"]
        @test pyis(jl2py(col), getfield(col, :py))
    end
end

@testset "keyword arguments are forwarded to python methods" begin
    # `train_test_split` requires the `test_size` keyword: regression test for the
    # kwargs-forwarding bug where keywords were splatted as positional arguments.
    split = mnist.train_test_split(test_size=0.2)
    @test split isa DatasetDict
    @test Set(keys(split)) == Set(["train", "test"])
    @test length(split["train"]) + length(split["test"]) == length(mnist)
end

@testset "raw Python observations via set_format!(ds, nothing)" begin
    # Passing `nothing` strips all formatting, opting out of the default "julia" format.
    mnist_raw = set_format!(copy(mnist), nothing)
    glue_raw = set_format!(copy(glue_ax), nothing)

    @test_throws BoundsError mnist_raw[0]
    @test length(mnist_raw[:]["label"]) == 10000

    x = mnist_raw[1]
    @test @py isinstance(x, dict)
    @test @py isinstance(x["image"], PIL.PngImagePlugin.PngImageFile)
    @test @py x["label"] === 7

    x = mnist_raw[1:2]
    @test @py isinstance(x, dict)
    @test @py isinstance(x["image"], list)
    @test @py isinstance(x["label"], list)
    @test @py isinstance(x["image"][1], PIL.PngImagePlugin.PngImageFile)
    @test Bool(@py x["label"] == [7, 2])

    x = glue_raw[1]
    @test Bool(@py x == {"premise": "The cat sat on the mat.",
                        "idx": 0,
                        "hypothesis": "The cat did not sit on the mat.",
                        "label": -1})

    x = glue_raw[1:2]
    @test @py isinstance(x["premise"], list)
    @test length(x["premise"]) == 2

    # the originals are untouched by the copy-on-write reset
    @test mnist.format["type"] == "numpy"
    @test glue_ax.format["type"] == "numpy"
end

@testset "with_format(julia) - mnist" begin
    ds = with_format(mnist, "julia")
    @test ds.format["type"] == "numpy"  # the julia format is numpy + py2jl under the hood

    # `ds[column]` returns a lazy `Column` view, not a materialized vector
    col = ds["label"]
    @test col isa HuggingFaceDatasets.Column{Int}
    @test col isa AbstractVector{Int}
    @test length(col) == 10000
    @test col[1] == 7
    @test col[1:2] == [7, 2]
    @test col[[2, 1]] == [2, 7]
    @test collect(col) isa Vector{Int}
    @test length(collect(col)) == 10000

    # under the numpy format an Image feature decodes to a real array (no PIL round-trip),
    # so a single row is an `(H, W)` matrix and a range stacks into an `(H, W, N)` tensor
    x = ds[1]
    @test x isa Dict
    @test x["label"] == 7
    @test x["image"] isa AbstractMatrix{UInt8}
    @test size(x["image"]) == (28, 28)

    x = ds[1:2]
    @test x isa Dict
    @test x["label"] isa Vector{Int}
    @test x["label"] == [7, 2]
    @test x["image"] isa AbstractArray{UInt8, 3}
    @test size(x["image"]) == (28, 28, 2)
    @test x["image"][:, :, 1] == ds[1]["image"]
end

@testset "with_format(julia) - glue_ax" begin
    ds = with_format(glue_ax, "julia")
    @test ds.format["type"] == "numpy"

    x = ds[1]
    @test x isa Dict
    @test x["label"] == -1
    @test x["idx"] == 0
    @test x["premise"] isa AbstractString
    @test x["premise"] == "The cat sat on the mat."
    @test x["hypothesis"] isa AbstractString
    @test x["hypothesis"] == "The cat did not sit on the mat."

    @test length(x["premise"]) == 23
    @test x["premise"][1] == 'T'

    x = ds[1:2]
    @test x isa Dict
    @test x["label"] isa Vector{Int}
    @test x["label"] == [-1, -1]
    @test x["idx"] == [0, 1]
    @test x["premise"] isa AbstractVector{<:AbstractString}
    @test all(x["premise"] .== ["The cat sat on the mat.", "The cat did not sit on the mat."])
    @test x["hypothesis"] isa AbstractVector{<:AbstractString}
    @test all(x["hypothesis"] .== ["The cat did not sit on the mat.", "The cat sat on the mat."])
end

@testset "python transforms" begin
    @pyexec ```
    def pytr(x):
        return {"label": [-l for l in x["label"]]}
    ``` => pytr

    ds = mnist.with_transform(pytr)
    
    @test Bool(ds[1]["label"] == -7)

    # format resets the python transform
    ds = with_format(ds, "julia")
    ds[1]["label"] == 7

    # python transforms and jltransforms can be composed
    ds = mnist.with_transform(pytr)
    ds = with_jltransform(ds, x -> py2jl(x["label"]) .- 1)
    @test ds[1:2] isa Vector{Int}
    @test ds[1:2] == [-8, -3]
end

@testset "reset_format! restores the default julia format" begin
    ds = set_format!(copy(mnist), nothing)    # start from raw Python
    @test ds.format["type"] === nothing
    @test @py isinstance(ds[1], dict)
    reset_format!(ds)                          # reset -> default "julia" format
    @test ds.format["type"] == "numpy"
    @test ds[1] isa Dict
    @test ds[1]["label"] == 7
end

@testset "set_format" begin
    ds = deepcopy(mnist)
    set_format!(ds, nothing)
    @test ds.format["type"] === nothing         # the copy is stripped ...
    @test mnist.format["type"] == "numpy"        # ... the original julia format is intact
    ds.set_format("numpy")
    @test ds.format["type"] == "numpy"
    set_format!(ds, nothing)
    @test ds.format["type"] === nothing
end

@testset "property-style format methods (python interface)" begin
    # `ds.with_format`/`ds.set_format`/`ds.reset_format` route to this package's methods
    # (not raw Python forwarding), so the `"julia"` pseudo-format works through them and
    # stays consistent with the function forms.
    ds = deepcopy(mnist)
    set_format!(ds, nothing)
    @test ds.format["type"] === nothing

    ds.set_format("julia")                       # would error if forwarded to Python
    @test ds.format["type"] == "numpy"           # julia format is numpy-backed
    @test ds[1] isa Dict && ds[1]["label"] == 7

    set_format!(ds, nothing)
    ds.reset_format()                            # restores julia, not Python's strip-to-raw
    @test ds.format["type"] == "numpy"
    @test ds[1] isa Dict

    dsj = ds.with_format("julia")                # non-mutating; returns a wrapped Dataset
    @test dsj isa Dataset
    @test dsj[1]["label"] == 7
end

@testset "Dataset.from_dict (python classmethod name)" begin
    # `Dataset.from_dict` mirrors `datasets.Dataset.from_dict` via type-level getproperty,
    # matching the `Dataset(data)` constructor (orientation-aware columns, "julia" format).
    ds = Dataset.from_dict(Dict("label" => [5, 0, 4]))
    @test ds isa Dataset
    @test ds.format["type"] == "numpy"           # default julia format, like the constructor
    @test ds[1] == Dict("label" => 5)
    @test ds[1:3]["label"] == [5, 0, 4]

    # NamedTuple input, and parity with the `Dataset(data)` constructor
    @test Dataset.from_dict((; label = [5, 0, 4]))[1:3]["label"] == Dataset((; label = [5, 0, 4]))[1:3]["label"]

    # N-D array column uses the same last-axis stacking as the constructor
    m = [1 2 3; 4 5 6]
    a = Dataset.from_dict((; x = m))
    @test a[1]["x"] == [1, 4]
    @test a[1:3]["x"] == m

    # a raw Python mapping is accepted too (converted via jl2py)
    c = Dataset.from_dict(pydict(Dict("label" => pylist([10, 20]))))
    @test c[2]["label"] == 20

    # type introspection is unaffected by the type-level getproperty overload
    @test fieldnames(Dataset) == (:py, :jltransform)
end

@testset "jltransform always acts on batches" begin
    ds = with_jltransform(mnist) do x
        x = py2jl(x)
        @test x["label"] isa Vector # batch, even when indexing with a single integer
        return x
    end
    ds[1]
    ds[1:2]
end

@testset "julia format survives forwarded methods" begin
    ds = Dataset(HuggingFaceDatasets.datasets.Dataset.from_dict(
        pydict(Dict("label" => pylist([5, 0, 4, 3, 2, 1])))))
    dsj = with_format(ds, "julia")

    # Methods returning a new `Dataset` keep the `"julia"` format (item 3 regression).
    @test dsj.shuffle(seed=0)[1] isa Dict{String, Int64}
    @test dsj.select(0:1)[1] isa Dict{String, Int64}
    @test dsj.filter(@pyeval("lambda x: x['label'] > 2"))[1] isa Dict{String, Int64}

    # ... and so does `train_test_split`, whose result is a `DatasetDict`.
    split = dsj.train_test_split(test_size=0.5)
    @test split isa DatasetDict
    @test split["train"][1] isa Dict{String, Int64}

    # A custom `jltransform` is propagated too.
    dst = with_jltransform(ds, x -> py2jl(x["label"]) .+ 100)
    @test dst.shuffle(seed=0)[1:2] isa Vector{Int}
    @test all(≥(100), dst.select(0:1)[1:2])

    # For a non-"julia" Python format, the transform is `identity`, so re-attaching is a
    # no-op and Python's own format propagation still governs.
    dsn = with_format(ds, "numpy")
    @test getfield(dsn.select(0:1), :jltransform) === identity
    @test dsn.select(0:1).format["type"] == "numpy"
end

@testset "julia-friendly map / filter" begin
    ds = Dataset(HuggingFaceDatasets.datasets.Dataset.from_dict(
        pydict(Dict("label" => pylist([5, 0, 4, 3, 2, 1])))))
    dsj = with_format(ds, "julia")

    # `map` bridges Julia values on both sides: pure-Julia callback, no PythonCall dialect.
    ds2 = map(x -> Dict("label" => x["label"] .+ 100), dsj; batched=true)
    @test ds2 isa Dataset
    @test ds2[1:6]["label"] == [105, 100, 104, 103, 102, 101]  # julia format preserved

    # non-batched map (one example at a time)
    ds3 = map(x -> Dict("label" => x["label"] + 1), dsj)
    @test ds3[1:6]["label"] == [6, 1, 5, 4, 3, 2]

    # `filter` with a Julia predicate returning a Bool
    ds4 = filter(x -> x["label"] > 2, dsj)
    @test ds4 isa Dataset
    @test sort(ds4[1:length(ds4)]["label"]) == [3, 4, 5]

    # batched filter returning a Vector{Bool}, and format preservation
    ds5 = filter(x -> x["label"] .< 3, dsj; batched=true)
    @test ds5[1] isa Dict{String, Int64}
    @test sort(ds5[1:length(ds5)]["label"]) == [0, 1, 2]

    # property-style `ds.map` / `ds.filter` route to the julia-bridged versions above
    # (python interface), i.e. `ds.map(f)` == `map(f, ds)` — the callback still sees Julia
    # values. A raw Python callback would go through `ds.py.map(...)` instead.
    dm = dsj.map(x -> Dict("label" => x["label"] .+ 100); batched=true)
    @test dm isa Dataset
    @test dm[1:6]["label"] == [105, 100, 104, 103, 102, 101]
    df = dsj.filter(x -> x["label"] > 2)
    @test df isa Dataset
    @test sort(df[1:length(df)]["label"]) == [3, 4, 5]
end
