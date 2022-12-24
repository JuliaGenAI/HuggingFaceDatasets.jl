using HuggingFaceDatasets
using Test

@testset "d.with_transform(julia) - mnist" begin
    d = load_dataset("mnist", split="test").with_format("julia")
    @test_throws AssertionError d[0]
    @test_throws AssertionError d[[3, -1]]
    @test length(d) == 10000
    @test d[1] isa Dict
    @test d[1]["image"] isa Matrix{UInt8}
    @test size(d[1]["image"]) == (28, 28)
    @test d[1]["label"] isa Int
    @test d[1]["label"] == 7

    @test d[1:2] isa Dict
    @test d[1:2]["image"] isa Array{UInt8,3}
    @test size(d[1:2]["image"]) == (28, 28, 2)
end

@testset "d.with_format(julia) - glue" begin
    d = load_dataset("glue", "sst2", split="train").with_format("julia")
    @test d[1] isa Dict
    @test d[1]["sentence"] isa AbstractString
    @test d[1]["label"] isa Int
    @test d[1]["label"] == 0
    @test d[1]["idx"] == 0
    @test d[1]["sentence"] == "hide new secretions from the parental units "
    
    # https://github.com/cjdoris/PythonCall.jl/issues/254
    @test_broken length(d[1]["sentence"])
    @test_broken d[1]["sentence"][1] == ["h"]

    @test d[1:2] isa Dict
    @test d[1:2]["sentence"] isa AbstractVector{<:AbstractString}
    @test d[1:2]["sentence"] isa PyVector{<:AbstractString}
end
