
@testset "mnist" begin
    d = load_dataset("mnist", split="test")
    
    @testset "load_dataset" begin
        @test d isa Dataset
        @test length(d) == 10000
    end

    @testset "indexing with no transform" begin
        tr = d.transform
        set_transform!(d, identity)
        
        @test_throws AssertionError d[0]
        @test d[1] isa Py
        @test pyisinstance(d[1], pytype(pydict()))
        @test d[1]["image"] isa Py
        @test d[1]["label"] isa Py
        @test pyisinstance(d[1]["label"], pytype(pyint()))
        @test py2jl(d[1]["label"]) == 7
        @test py2jl(d[2]["label"]) == 2
        
        @test d[1:2] isa Py
        @test d[1:2]["image"] isa Py
        @test pyisinstance(d[1:2]["image"], pytype(pylist()))
        @test d[1:2]["label"] isa Py
        @test pyisinstance(d[1:2]["label"], pytype(pylist()))
        
        set_transform!(d, tr)
    end

    @testset "indexing - py2jl" begin
        @test d.transform === py2jl
        sample = d[1]
        @test sample isa Dict
        @test sample["label"] isa Int
        @test sample["label"] == 7
        @test sample["image"] isa Matrix{UInt8}
        @test size(sample["image"]) == (28, 28)

        sample = d[1:2]
        @test sample isa Dict
        @test sample["image"] isa Vector{Matrix{UInt8}}
        @test size(sample["image"]) == (2,)
        @test sample["label"] isa Vector{Int}
        @test size(sample["label"]) == (2,)
    end

    @testset "python transforms" begin
        @pyexec """
        def pytr(x):
            return {"label": [-l for l in x["label"]]}
        """ => pytr
        d.set_transform(pytr)
        @test d[1]["label"] == -7
    end
end
