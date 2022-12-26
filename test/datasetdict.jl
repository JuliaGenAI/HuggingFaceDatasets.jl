
mnist = load_dataset("mnist")

@test mnist isa DatasetDict
@test length(mnist) == 2

@testset "indexing, no (jl)transform by default" begin
    @test_throws MethodError mnist[1]
    @test mnist["test"] isa Dataset
    ds = mnist["test"]
    @test pyisinstance(ds[1], pytype(pydict()))
    @test ds[1]["image"] isa Py
    @test ds[1]["label"] isa Py
    @test pyisinstance(ds[1]["label"], pytype(pyint()))
    @test py2jl(ds[1]["label"]) == 7
    @test py2jl(ds[2]["label"]) == 2
end

@testset "with_format(julia)" begin
    d = with_format(mnist, "julia")
    ds = d["test"]
    @test ds.format["type"] == "numpy"
    x = ds[1]
    @test x isa Dict
    @test x["label"] isa Int
    @test x["label"] == 7
    @test x["image"] isa Matrix{UInt8}
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
    d.set_format("numpy")
    @test d["test"].format["type"] == "numpy"
    @test mnist["test"].format["type"] === nothing
    set_format!(d, nothing)
    @test d["test"].format["type"] === nothing
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
