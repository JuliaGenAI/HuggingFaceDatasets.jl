@testset "transform" begin
    d = load_dataset("mnist", split="test")
    @test d.transform === identity
end
