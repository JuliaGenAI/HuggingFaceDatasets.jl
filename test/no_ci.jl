using HuggingFaceDatasets, ImageShow

@testset "image classification" begin 
    @testset "cifar10" begin
        ds = load_dataset("cifar10", split = "test").with_format("julia")
        @test ds[1]["img"] isa AbstractMatrix{RGB{N0f8}}
        @test ds[1]["label"] isa Int

        @test ds[1:2]["img"] isa Vector{<:AbstractMatrix{RGB{N0f8}}}
        @test ds[1:2]["label"] isa Vector{Int}
    end

    @testset "beans" begin
        ds = load_dataset("beans", split = "test").with_format("julia")
        @test ds[1]["image"] isa AbstractMatrix{RGB{N0f8}}
        @test ds[1]["labels"] isa Int

        @test ds[1:2]["image"] isa Vector{<:AbstractMatrix{RGB{N0f8}}}
        @test ds[1:2]["labels"] isa Vector{Int}
    end
end

@testset "object detection" begin
    @testset "cppe-5" begin
        ds = load_dataset("cppe-5", split = "test").with_format("julia")
        @test ds[1]["image"] isa AbstractMatrix{RGB{N0f8}}
        @test ds[1]["objects"] isa Dict{String, Vector}

        @test ds[1:2]["image"] isa Vector{<:AbstractMatrix{RGB{N0f8}}}
        @test ds[1:2]["objects"] isa Vector{Dict{String, Vector}}
    end
end
