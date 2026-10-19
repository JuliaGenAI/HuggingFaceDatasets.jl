using HuggingFaceDatasets: _content_hash, _Unhashable

fp(f; kws...) = _content_hash(f, pairs((; kws...)))

@testset "what the content hash of a transform follows" begin
    m = Module()
    @eval m begin
        k = [1]
        g(x) = x + k[1]
        f(b) = Dict("y" => map(x -> g(x) + 1, b["x"]))
    end
    h = fp(m.f)
    @test h isa String
    @test fp(m.f) == h

    # the same code in another module, with other generated closure names, hashes the same
    m2 = Module()
    @eval m2 begin
        shift = [x -> x + i for i in 1:3]
        k = [1]
        g(x) = x + k[1]
        f(b) = Dict("y" => map(x -> g(x) + 1, b["x"]))
    end
    @test fp(m2.f) == h

    # a global it reads, a function it calls, an inner closure
    m.k[1] = 2
    @test fp(m.f) != h
    m.k[1] = 1
    @test fp(m.f) == h
    @eval m g(x) = x - k[1]
    @test fp(m.f) != h
    @eval m g(x) = x + k[1]
    @test fp(m.f) == h
    @eval m f(b) = Dict("y" => map(x -> g(x) + 2, b["x"]))
    @test fp(m.f) != h

    # a keyword argument's default
    @eval m kw(x; s = 2) = x * s
    @eval m u(x) = kw(x)
    hu = fp(m.u)
    @eval m kw(x; s = 3) = x * s
    @test fp(m.u) != hu

    # captured values, and the methods of a callable struct
    @eval m mk(c) = b -> b .+ c
    @test fp(m.mk(1)) == fp(m.mk(1)) != fp(m.mk(2))
    @eval m begin
        struct P
            a::Int
        end
        (p::P)(x) = x + p.a
    end
    hp = fp(m.P(1))
    @test hp != fp(m.P(2))
    @eval m (p::P)(x) = x * p.a
    @test fp(m.P(1)) != hp

    # a function of a submodule, called qualified
    @eval m module U
        t(x) = x + 1
    end
    @eval m r(x) = U.t(x)
    hr = fp(m.r)
    @eval m.U t(x) = x + 2
    @test fp(m.r) != hr

    # a method of your own on another module's function that the code calls
    @eval m begin
        struct Q
            v::Int
        end
        Base.length(q::Q) = q.v
        s(q) = length(q)
    end
    hs = fp(m.s)
    @eval m Base.length(q::Q) = 2 * q.v
    @test fp(m.s) != hs

    # keyword arguments, except those `datasets` leaves out of its fingerprint
    @test fp(m.f; batched = true) != fp(m.f; batched = false)
    @test fp(m.f; batched = true, desc = "d", load_from_cache_file = false) == fp(m.f; batched = true)
    @test fp(m.f; fn_kwargs = Dict("a" => 1)) != fp(m.f; fn_kwargs = Dict("a" => 2))

    # captured data, compared by value
    @eval m capture(x) = b -> x
    d = Dict(:a => 1, :b => [1.0, 2.0])
    hd = fp(m.capture(d))
    @test fp(m.capture(Dict(:b => [1.0, 2.0], :a => 1))) == hd
    d[:b][1] = 3.0
    @test fp(m.capture(d)) != hd
    @test fp(m.capture(r"a+")) != fp(m.capture(r"b+"))
    @test fp(m.capture(big(2)^70)) != fp(m.capture(big(2)^71))
    @test fp(m.capture(Float32[1, 2])) != fp(m.capture(Float32[1, 3]))

    # Python objects, hashed by `datasets`
    builtins = pyimport("builtins")
    @test fp(m.capture(builtins.len)) isa String
    @test fp(m.capture(builtins.len)) != fp(m.capture(builtins.abs))
end

@testset "transforms that can't be hashed" begin
    m = Module()
    @eval m capture(x) = b -> x
    @test fp(m.capture(Channel(1))) isa _Unhashable
    @test fp(m.capture(Ptr{Int}(1))) isa _Unhashable
    @test fp(m.capture(Dataset((; x = [1, 2])).to_iterable_dataset())) isa _Unhashable
    @eval m @generated gen(x) = :(x)
    @test fp(m.gen) isa _Unhashable
end
