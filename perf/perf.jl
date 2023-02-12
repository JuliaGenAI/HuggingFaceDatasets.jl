using HuggingFaceDatasets
using BenchmarkTools
using MLDatasets

function f(ds)
    for i in 1:numobs(ds)
        getobs(ds, i)
    end
end
function fbatch(ds)
    for i in 1:128:numobs(ds)-128
        getobs(ds, i:i+127)
    end
end
function fall(ds)
    getobs(ds, :)
end

function bench()
    mld = MNIST(split=:test)
    ds_plain = load_dataset("mnist", split="test")
    ds_julia = with_format(ds_plain, "julia")
    ds_numpy = with_format(ds_plain, "numpy")
    ds_jnumpy = with_jltransform(py2jl, ds_numpy) # numpy + py2jl

    for (name, ds) in [("mldatasets", mld),
                      ("plain", ds_plain),
                      ("julia", ds_julia),
                      ("numpy", ds_numpy),
                      ("jnumpy", ds_jnumpy)]
        println("# $name")
        @btime f($ds)
        @btime fbatch($ds)
        @btime fall($ds)
    end
end

# hf is slow at reading image datasets.
# Pytorch vision is much faset (see the notebook in perf/) 

bench()
# # MLDatasets
# 19.515 ms (120005 allocations: 34.64 MiB)
# 4.671 ms (1097 allocations: 29.97 MiB)
# 717.324 ns (6 allocations: 240 bytes)
# # plain
# 602.001 ms (668464 allocations: 18.06 MiB)
# 266.483 ms (390 allocations: 6.09 KiB)
# 265.651 ms (5 allocations: 80 bytes)
# # julia
# 985.251 ms (2398464 allocations: 93.28 MiB)
# 379.270 ms (659256 allocations: 27.31 MiB)
# 378.751 ms (650134 allocations: 27.01 MiB)
# # numpy
# 1.264 s (728464 allocations: 19.13 MiB)
# 311.426 ms (390 allocations: 6.09 KiB)
# 318.403 ms (5 allocations: 80 bytes)
# # jnumpy
# 1.527 s (2208464 allocations: 110.91 MiB)
# 318.356 ms (13962 allocations: 637.41 KiB)
# 335.109 ms (179 allocations: 8.17 KiB)
