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
    @test col[3:-1:1] == [4, 0, 5]             # ranges are read as a Python `range`
    @test col[Base.OneTo(2)] == [5, 0]
    @test isempty(col[1:0])
    @test col == [5, 0, 4]
    @test collect(col) isa Vector{Int}
    @test collect(col) == [5, 0, 4]

    # a 0-d numpy array (produced by the numpy formatter when it tensorizes a plain scalar
    # cell) unwraps to a native scalar, not a `fill(x)` 0-d Julia array
    z = np.asarray(7)
    @test py2jl(z) == 7
    @test py2jl(z) isa Integer

    # numeric numpy scalars (and 0-d arrays) keep their dtype instead of widening to Int64/Float64
    for (name, T) in (("int8", Int8), ("int32", Int32), ("uint16", UInt16), ("uint64", UInt64),
                      ("float16", Float16), ("float32", Float32), ("float64", Float64),
                      ("complex64", ComplexF32), ("longlong", Int64))
        s = pygetattr(np, name)(3)
        @test py2jl(s) === T(3)
        @test py2jl(np.asarray(s)) === T(3)
    end
    @test py2jl(np.bool_(true)) === true
    @test py2jl(np.asarray(true)) === true
    @test py2jl(np.str_("ab")) == "ab"
    @test py2jl(np.asarray("ab")) == "ab"
end

@testset "jl2py" begin
    # other scalars fall back to PythonCall's default conversion: Bool, Int64 and Float64
    # become Python scalars
    @test pyconvert(Int, jl2py(3)) == 3
    @test pyconvert(String, jl2py("a")) == "a"
    @test pyisinstance(jl2py(true), pybuiltins.bool)
    @test pyisinstance(jl2py(3), pybuiltins.int)
    @test pyisinstance(jl2py(2.5), pybuiltins.float)

    # narrow numeric scalars -> numpy scalars keeping their dtype (#93)
    for x in (Int8(-7), Int16(-7), Int32(-7), UInt8(7), UInt16(7), UInt32(7), typemax(UInt64),
              Float16(1.5), Float32(1.5))
        y = jl2py(x)
        @test pyisinstance(y, np.generic)
        @test pyconvert(String, y.dtype.name) == lowercase(string(typeof(x)))
        @test pyconvert(typeof(x), y) === x
    end

    # ... also nested in containers
    y = jl2py((Int8(1), Dict("a" => Float32(2)), Any[UInt16(3), "b"]))
    @test pyconvert(String, y[0].dtype.name) == "int8"
    @test pyconvert(String, y[1]["a"].dtype.name) == "float32"
    @test pyconvert(String, y[2][0].dtype.name) == "uint16"
    # ... and as dict keys, which hash like the equal Python int
    @test pyconvert(String, jl2py(Dict(Int32(1) => "a"))[1]) == "a"

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
    @test py2jl(jl2py(x')) == x'
    # ... on a copy, unlike jl2numpy itself (#94)
    y = jl2py(x)
    x[1, 1] = 100
    @test pyconvert(Float32, y[0, 0]) == 1

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