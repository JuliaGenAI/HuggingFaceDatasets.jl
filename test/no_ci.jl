# Under the "julia" (= numpy) format, Image features decode to raw numeric arrays rather
# than `RGB`/`Gray` colorviews: the numpy format materializes them before `py2jl` sees a
# PIL object. The axes are reversed by `numpy2jl`, so an `(H, W, C)` image becomes a
# `(C, W, H)` Julia array; a range index stacks fixed-shape images along the last axis and
# falls back to a vector-of-arrays for ragged (variable-size) image columns.
@testset "image classification" begin
    @testset "cifar10" begin
        ds = load_dataset("uoft-cs/cifar10", split = "test").with_format("julia")
        @test ds[1]["img"] isa Array{UInt8, 3}
        @test size(ds[1]["img"]) == (3, 32, 32)       # (C, W, H)
        @test ds[1]["label"] isa Int

        b = ds[1:2]["img"]
        @test b isa Array{UInt8, 4}
        @test size(b) == (3, 32, 32, 2)               # stacked, observation axis last
        @test b[:, :, :, 1] == ds[1]["img"]
        @test ds[1:2]["label"] isa Vector{Int}
    end

    @testset "beans" begin
        ds = load_dataset("AI-Lab-Makerere/beans", split = "test").with_format("julia")
        @test ds[1]["image"] isa Array{UInt8, 3}
        @test size(ds[1]["image"]) == (3, 500, 500)
        @test ds[1]["labels"] isa Int

        b = ds[1:2]["image"]
        @test b isa Array{UInt8, 4}
        @test size(b) == (3, 500, 500, 2)
        @test ds[1:2]["labels"] isa Vector{Int}
    end
end

@testset "object detection" begin
    @testset "cppe-5" begin
        ds = load_dataset("rishitdagli/cppe-5", split = "test").with_format("julia")
        @test ds[1]["image"] isa Array{UInt8, 3}
        @test size(ds[1]["image"]) == (3, 1920, 1088)
        @test ds[1]["objects"] isa Dict{String, <:Any}

        # cppe-5 images have varying sizes, so a batch cannot numpy-stack and falls back
        # to a vector of per-observation arrays.
        imgs = ds[1:2]["image"]
        @test imgs isa Vector{<:Array{UInt8, 3}}
        @test imgs[1] == ds[1]["image"]
        @test ds[1:2]["objects"] isa Vector{<:Dict{String}}
    end
end
