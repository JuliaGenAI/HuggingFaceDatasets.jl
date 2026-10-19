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

    # a 0-d numpy array (produced by the numpy formatter when it tensorizes a plain scalar
    # cell) unwraps to a native scalar, not a `fill(x)` 0-d Julia array
    z = np.asarray(7)
    @test py2jl(z) == 7
    @test py2jl(z) isa Integer
end

@testset "jl2py" begin
    # scalars fall back to PythonCall's default conversion
    @test pyconvert(Int, jl2py(3)) == 3
    @test pyconvert(String, jl2py("a")) == "a"

    # a Py passes through unchanged
    p = pylist([1, 2])
    @test jl2py(p) === p

    # numeric vectors -> 1-D numpy arrays keeping their element type (#80)
    for T in (Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
              Float16, Float32, Float64)
        x = T[1, 0, 1]
        y = jl2py(x)
        @test pyisinstance(y, np.ndarray)
        @test pyconvert(String, y.dtype.name) == lowercase(string(T))
        @test py2jl(y) == x
    end

    # lazy / non-contiguous numeric vectors are materialized to a contiguous array
    v = collect(Int16, 1:6)
    for x in Any[Int16(1):Int16(3), view(v, 1:2:5), view(v, 2:4), reinterpret(UInt8, v),
                 trues(3), Int32[]]
        y = jl2py(x)
        @test pyisinstance(y, np.ndarray)
        @test pyconvert(String, y.dtype.name) == lowercase(string(eltype(x)))
        @test py2jl(y) == collect(x)
    end

    # the vector is copied, not shared: later mutations don't reach the numpy array, and
    # the array stays valid once the Julia vector is garbage-collected
    x = [1.0, 2.0, 3.0]
    y = jl2py(x)
    x[1] = 99.0
    @test pyconvert(Float64, y[0]) == 1.0
    y = let x = collect(1:10_000)
        jl2py(x)
    end
    GC.gc(true)
    @test pyconvert(Int, y.sum()) == sum(1:10_000)

    # other vectors -> python lists (element-wise): no numpy dtype (or no Arrow type) exists
    for x in Any[["a", "b"], Any[1, 2.5], Union{Missing, Int}[1, missing],
                 Union{Int32, Int64}[1, 2], BigInt[1, 2], [1//2, 3//4], Int128[1],
                 ComplexF64[1 + 2im], Union{}[]]
        @test pyisinstance(jl2py(x), pytype(pylist()))
    end

    # dicts/namedtuples -> python dicts
    @test py2jl(jl2py(Dict("a" => [1, 2], "b" => [3, 4]))) == Dict("a" => [1, 2], "b" => [3, 4])
    @test py2jl(jl2py((label = [1, 2], x = [3, 4]))) == Dict("label" => [1, 2], "x" => [3, 4])
    @test py2jl(jl2py((1, "a", [2, 3]))) == (1, "a", [2, 3])

    # N-D numeric arrays go through jl2numpy (axis-reversed, round-trips via py2jl)
    x = Float32[1 2 3; 4 5 6]
    @test py2jl(jl2py(x)) == x

    # a vector of arrays becomes a python list of numpy arrays
    v = [Int[1, 2], Int[3, 4]]
    @test py2jl(jl2py(v)) == v
    y = jl2py([UInt16[1, 2], UInt16[3]])
    @test pyisinstance(y, pytype(pylist()))
    @test pyconvert(String, y[0].dtype.name) == "uint16"
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