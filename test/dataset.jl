mnist = load_dataset("mnist", split="test")
glue_ax = load_dataset("glue", "ax", split="test")

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
    @test mnist.format["type"] === nothing
end

@testset "indexing, no (jl)transform by default" begin
    @test_throws AssertionError mnist[0]
    
    x = mnist[1]
    @test @py isinstance(x, dict)
    @py isinstance(x["image"], PIL.PngImagePlugin.PngImageFile)
    @test @py x["label"] === 7

    x = mnist[1:2]
    @test @py isinstance(x, dict)
    @test @py isinstance(x["image"], list)
    @test @py isinstance(x["label"], list)
    @test @py isinstance(x["image"][1], PIL.PngImagePlugin.PngImageFile)
    @test Bool(@py x["label"] == [7, 2])

    x = glue_ax[1]
    @test Bool(@py x == {"premise": "The cat sat on the mat.", 
                        "idx": 0, 
                        "hypothesis": "The cat did not sit on the mat.", 
                        "label": -1})

    x = glue_ax[1:2]
    @test @py isinstance(x["premise"], list)
    @test length(x["premise"]) == 2
end

@testset "with_format(julia) - mnist" begin
    ds = with_format(mnist, "julia")
    @test ds.format["type"] === "numpy"

    x = ds[1]
    @test x isa Dict
    @test x["label"] == 7
    @test x["image"] isa Matrix{UInt8}
    @test size(x["image"]) == (28, 28)

    x = ds[1:2]
    @test x isa Dict
    @test x["label"] isa Vector{Int}
    @test x["label"] == [7, 2]
    @test x["image"] isa Array{UInt8,3}
    @test size(x["image"]) == (28, 28, 2)
end

@testset "with_format(julia) - glue_ax" begin
    ds = with_format(glue_ax, "julia")
    @test ds.format["type"] === "numpy"

    x = ds[1]
    @test x isa Dict
    @test x["label"] == -1
    @test x["idx"] == 0
    @test x["premise"] isa AbstractString
    @test x["premise"] == "The cat sat on the mat."
    @test x["hypothesis"] isa AbstractString
    @test x["hypothesis"] == "The cat did not sit on the mat."

    # https://github.com/cjdoris/PythonCall.jl/issues/254
    @test_broken length(x["premise"])
    @test_broken x["sentence"][1] == ["h"]

    x = ds[1:2]
    @test x isa Dict
    @test x["label"] isa Vector{Int}
    @test x["label"] == [-1, -1]
    @test x["idx"] == [0, 1]
    @test x["premise"] isa AbstractVector{<:AbstractString}
    @test x["premise"] isa PyArray{<:AbstractString, 1}
    @test all(x["premise"] .== ["The cat sat on the mat.", "The cat did not sit on the mat."])
    @test x["hypothesis"] isa AbstractVector{<:AbstractString}
    @test x["hypothesis"] isa PyArray{<:AbstractString, 1}
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

@testset "reset_format!" begin
    ds = with_format(mnist, "julia")
    @test ds.format["type"] == "numpy"
    reset_format!(ds)
    @test ds.format["type"] === nothing
    @test @py isinstance(ds[1], dict)
end
