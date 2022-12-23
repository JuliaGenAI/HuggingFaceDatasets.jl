
@testset "MNIST" begin
    dd = load_dataset("mnist")
    
    @testset "load_dataset" begin
        @test dd isa DatasetDict
        @test length(dd) == 2
    end

    @testset "indexing with no transform" begin
        tr = dd.transform
        set_transform!(dd, identity)
        
        @test_throws MethodError dd[1]
        @test dd["test"] isa Dataset
        d = dd["test"]
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
        
        set_transform!(dd, tr)
    end

    @testset "indexing - py2jl" begin
        @test dd.transform === py2jl
        d = dd["test"]
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
        dd.set_transform(pytr)
        @test dd["test"][1]["label"] == -7
    end

    @testset "getproperty returns julia types" begin
        @test dd.num_rows isa Dict{String, Int}
        @test dd.num_rows == Dict("test"  => 10000, "train" => 60000)
    end
end

