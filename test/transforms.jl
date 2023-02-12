@testset "py2jl" begin
    l = pylist([1,2,3])
    @test py2jl(l) isa Vector{Int}

    d = pydict(["a" => 1, "b" => 2])
    @test py2jl(d) isa Dict{String,Int}
end

# only if no CI
# if isempty(ENV["CI"])
#     @testset "py2jl" begin
#         l = pylist([1,2,3])
#         @test py2jl(l) isa Vector{Int}

#         d = pydict(["a" => 1, "b" => 2])
#         @test py2jl(d) isa Dict{String,Int}
#     end
# end