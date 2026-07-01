@testset "py2jl" begin
    l = pylist([1,2,3])
    @test py2jl(l) isa Vector{Int}

    d = pydict(["a" => 1, "b" => 2])
    @test py2jl(d) isa Dict{String,Int}

    # tuples are converted element-wise to a Julia tuple (regression: used to return a
    # 1-tuple wrapping an unevaluated generator)
    t = py2jl(pytuple((1, pylist([2, 3]))))
    @test t isa Tuple
    @test t == (1, [2, 3])

    # a datasets.Column (datasets >= 4) is wrapped in a lazy `Column`, converting
    # elements on access rather than materializing the whole column
    pyds = datasets.Dataset.from_dict(pydict(label = [5, 0, 4]))
    col = py2jl(pyds["label"])
    @test col isa HuggingFaceDatasets.Column{Int}
    @test col isa AbstractVector{Int}
    @test size(col) == (3,)
    @test col[1] == 5
    @test col[2:3] == [0, 4]
    @test col[[3, 1]] == [4, 5]
    @test col[[true, false, true]] == [5, 4]   # logical indexing (Bool <: Integer)
    @test col == [5, 0, 4]
    @test collect(col) isa Vector{Int}
    @test collect(col) == [5, 0, 4]
end

@testset "jl2numpy / numpy2jl" begin
    # round-trip across dtypes and dimensionalities (dims are permuted then permuted back)
    for x in Any[Float32[1 2; 3 4], Int64[1, 2, 3, 4], rand(UInt8, 2, 3, 4)]
        y = jl2numpy(x)
        @test size(numpy2jl(y)) == size(x)
        @test numpy2jl(y) == x
    end

    # the numpy view shares memory with the Julia array in both directions
    x = [1.0 2.0; 3.0 4.0]
    y = jl2numpy(x)
    x[1, 1] = 99.0
    @test pyconvert(Float64, y[0, 0]) == 99.0   # julia -> numpy
    y[0, 0] = 7.0
    @test x[1, 1] == 7.0                         # numpy -> julia (writable)
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