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
    @test mnist.format["type"] == "julia"   # loaded in the "julia" format
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
    @test ds.format["type"] == "julia"                     # julia format by default
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
    @test ds[1:2]["f"] isa Vector{Float64}
    @test ds[1:2]["b"] == [true, false]

    # numeric columns keep their element type in the schema (#80)
    ds = Dataset((u8 = UInt8[1, 2], i32 = Int32[3, 4], f32 = Float32[1.5, 2.5],
                  f16 = Float16[1, 2], b = [true, false], r = 1:2))
    @test [ds.features[k].dtype for k in ds.column_names] ==
          ["uint8", "int32", "float32", "float16", "bool", "int64"]
    @test ds[1:2]["u8"] == [1, 2]
    @test ds[1:2]["i32"] == [3, 4]
    @test ds[1:2]["f32"] == [1.5, 2.5]
    @test ds[1:2]["f16"] == [1, 2]
    @test ds[1:2]["b"] == [true, false]
    @test ds[2]["r"] == 2
    @test pyconvert(String, pystr(Dataset((; x = [UInt16[1, 2], UInt16[3]])).py.features["x"])) ==
          "List(Value('uint16'))"
    @test Dataset((; x = Int[])).features["x"].dtype == "int64"

    # the column is copied: mutating the source vector afterwards leaves the dataset intact
    v = Float32[1, 2, 3]
    ds = Dataset((; v))
    v[1] = 100
    @test ds[1:3]["v"] == [1, 2, 3]

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

    # numeric columns keep their element type (#80)
    ds = Dataset([(u = 0x01, f = 1.5f0), (u = 0x02, f = 2.5f0)])
    @test ds.features["u"].dtype == "uint8"
    @test ds.features["f"].dtype == "float32"

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
        @test ds.format["type"] == "julia"          # default julia format
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
        @test ds.format["type"] == "julia"
        @test ds[:]["label"] == [1, 2, 1, 2, 1, 2]
        @test interleave_datasets([x, y])[:]["label"] == [1, 2, 1, 2, 1, 2]
    end

    @testset "save_to_disk / load_from_disk round-trip" begin
        mktempdir() do dir
            path = joinpath(dir, "ds")
            a.save_to_disk(path)                     # write side is a forwarded method
            # "julia" is saved as "numpy", so plain Python (without this package) can load it
            @test occursin("\"_format_type\": \"numpy\"", read(joinpath(path, "state.json"), String))
            @test a.format["type"] == "julia"
            loaded = load_from_disk(path)            # read side is the new top-level wrapper
            @test loaded isa Dataset
            @test loaded.format["type"] == "julia"   # default julia format
            @test loaded[:]["label"] == [1, 2]

            # the Python classmethod name also works, via type-level getproperty
            loaded = Dataset.load_from_disk(path)
            @test loaded isa Dataset
            @test loaded[:]["label"] == [1, 2]

            # ... and on DatasetDict for a saved dict of splits
            dpath = joinpath(dir, "dd")
            DatasetDict("train" => a).save_to_disk(dpath)
            @test occursin("\"_format_type\": \"numpy\"", read(joinpath(dpath, "train", "state.json"), String))
            dd = DatasetDict.load_from_disk(dpath)
            @test dd isa DatasetDict
            @test dd["train"][:]["label"] == [1, 2]
            @test load_from_disk(dpath) isa DatasetDict     # top-level auto-detects
        end
    end

    @testset "from_csv / from_json / from_parquet" begin
        mktempdir() do dir
            src = Dataset((label = [5, 0, 4], text = ["a", "b", "c"]))

            csv = joinpath(dir, "d.csv")
            src.py.to_csv(csv)
            ds = HuggingFaceDatasets.from_csv(csv)
            @test ds isa Dataset
            @test ds.format["type"] == "julia"
            @test ds[:]["label"] == [5, 0, 4]
            @test ds[:]["text"] == ["a", "b", "c"]
            @test Dataset.from_csv(csv)[:]["label"] == [5, 0, 4]   # classmethod name

            json = joinpath(dir, "d.json")
            src.py.to_json(json)
            @test HuggingFaceDatasets.from_json(json)[:]["label"] == [5, 0, 4]
            @test Dataset.from_json(json)[:]["label"] == [5, 0, 4]

            parquet = joinpath(dir, "d.parquet")
            src.py.to_parquet(parquet)
            @test HuggingFaceDatasets.from_parquet(parquet)[:]["label"] == [5, 0, 4]
            @test Dataset.from_parquet(parquet)[:]["label"] == [5, 0, 4]
        end
    end

    @testset "type introspection unaffected by type-level getproperty" begin
        @test fieldnames(Dataset) == (:py, :jltransform)
        @test fieldnames(DatasetDict) == (:py, :jltransform, :splits)
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
    @test mnist.format["type"] == "julia"
    @test glue_ax.format["type"] == "julia"
end

@testset "with_format(ds, nothing)" begin
    ds = with_format(mnist, nothing)
    @test ds isa Dataset
    @test ds.format["type"] === nothing
    x = ds[1]
    @test @py isinstance(x, dict)
    @test @py x["label"] === 7
    @test mnist.with_format(nothing).format["type"] === nothing   # property-style form
    # the original keeps the julia format
    @test mnist.format["type"] == "julia"
    @test mnist[1]["label"] == 7
end

@testset "with_format(julia) - mnist" begin
    ds = with_format(mnist, "julia")
    @test ds.format["type"] == "julia"  # the "julia" Python formatter + py2jl

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
    @test ds.format["type"] == "julia"

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

@testset "julia format keeps each column's dtype" begin
    # `datasets`' numpy formatter casts integers to int64 and floats to float32; the "julia"
    # formatter keeps the Arrow dtype (#78).
    V = datasets.Value
    feats = datasets.Features(pydict(Dict(
        "u16" => datasets.List(V("uint16")), "i8" => V("int8"), "i32" => V("int32"),
        "u64" => V("uint64"), "f16" => V("float16"), "f32" => V("float32"),
        "f64" => V("float64"), "b" => V("bool"), "rag" => datasets.List(V("int16")),
        "a2" => datasets.Array2D((2, 3), "float64"))))
    a2 = [rand(3, 2), rand(3, 2)]   # numpy (2, 3) after the axis reversal
    ds = Dataset.from_dict(Dict("u16" => [[1, 2], [3, 4]], "i8" => [1, -2], "i32" => [1, 2],
        "u64" => [1, 2], "f16" => [1.5, 2.5], "f32" => [1.5, 2.5], "f64" => [1.0000000001, 2.0],
        "b" => [true, false], "rag" => [[1, 2, 3], [4]], "a2" => a2); features = feats)

    b = ds[1:2]
    @test b["u16"] isa Matrix{UInt16} && b["u16"] == [1 3; 2 4]
    @test b["i8"] isa Vector{Int8} && b["i8"] == [1, -2]
    @test b["i32"] isa Vector{Int32}
    @test b["u64"] isa Vector{UInt64}
    @test b["f16"] isa Vector{Float16} && b["f16"] == [1.5, 2.5]
    @test b["f32"] isa Vector{Float32}
    @test b["f64"] isa Vector{Float64} && b["f64"] == [1.0000000001, 2.0]
    @test b["b"] isa Vector{Bool} && b["b"] == [true, false]
    @test b["rag"] isa Vector{Vector{Int16}} && b["rag"] == [[1, 2, 3], [4]]
    @test b["a2"] isa Array{Float64, 3} && b["a2"] == cat(a2...; dims = 3)

    r = ds[1]
    @test r["u16"] isa Vector{UInt16}
    @test r["i32"] isa Int32
    @test r["f64"] === 1.0000000001
    @test r["a2"] == a2[1]
    @test ds["f64"][1:2] isa Vector{Float64}   # lazy `Column` batches too

    # a Float64 array column built from Julia round-trips exactly
    x = rand(2, 3, 4)
    @test Dataset((; x))[1:4]["x"] == x

    # streaming reads go through the same formatter
    it = ds.to_iterable_dataset()
    r = first(it)
    @test r["u16"] isa Vector{UInt16}
    @test r["f64"] === 1.0000000001
    b = py2jl(first(it.py.iter(batch_size = 2)))
    @test b["u16"] isa Matrix{UInt16}
    @test b["f64"] isa Vector{Float64}

    # the plain "numpy" format is upstream's, casts included
    n = with_format(ds, "numpy")[1:2]
    @test pyconvert(String, n["u16"].dtype.name) == "int64"
    @test pyconvert(String, n["f64"].dtype.name) == "float32"
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
    @test ds.format["type"] == "julia"
    @test ds[1] isa Dict
    @test ds[1]["label"] == 7
end

@testset "set_format" begin
    ds = deepcopy(mnist)
    set_format!(ds, nothing)
    @test ds.format["type"] === nothing         # the copy is stripped ...
    @test mnist.format["type"] == "julia"        # ... the original julia format is intact
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
    @test ds.format["type"] == "julia"           # the Python format type is "julia" too
    @test ds[1] isa Dict && ds[1]["label"] == 7

    set_format!(ds, nothing)
    ds.reset_format()                            # restores julia, not Python's strip-to-raw
    @test ds.format["type"] == "julia"
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
    @test ds.format["type"] == "julia"           # default julia format, like the constructor
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
    ds6 = filter(x -> Vector{Bool}(x["label"] .> 3), dsj; batched=true)
    @test sort(ds6[1:length(ds6)]["label"]) == [4, 5]

    # numeric vectors returned by `f` keep their element type (#80)
    dv = map(b -> Dict("y" => [rand(UInt16, 4) for _ in b["label"]]), dsj;
             batched=true, batch_size=2)
    @test pyconvert(String, pystr(dv.py.features["y"])) == "List(Value('uint16'))"
    dv = map(b -> Dict("y" => Int32.(b["label"])), dsj; batched=true)
    @test dv.features["y"].dtype == "int32"
    @test dv[1]["y"] == 5                                    # still a scalar per row
    @test dv[1:6]["y"] == [5, 0, 4, 3, 2, 1]
    dv = map(x -> Dict("y" => Float32[x["label"], 1]), dsj)    # one vector per row
    @test pyconvert(String, pystr(dv.py.features["y"])) == "List(Value('float32'))"
    @test dv[1]["y"] == [5, 1]
    @test dv[1:2]["y"] == [5 0; 1 1]

    # each returned vector is copied: a buffer reused across calls doesn't alias the rows
    # that `datasets` holds before writing them
    buf = zeros(Int, 2)
    dv = map(x -> (buf .= x["label"]; Dict("y" => buf)), dsj)
    @test [dv[i]["y"] for i in 1:6] == [fill(l, 2) for l in [5, 0, 4, 3, 2, 1]]

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
