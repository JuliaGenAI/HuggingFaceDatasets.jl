@testset "transform" begin
    d = load_dataset("mnist", split="test")
    @test d.transform === py2jl

    d = load_dataset("mnist", split="test", transform=identity)
    @test d.transform === identity
end
