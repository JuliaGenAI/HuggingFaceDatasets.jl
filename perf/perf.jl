using HuggingFaceDatasets
using BenchmarkTools

function f(ds)
    for i in 1:6000
        ds[i]
    end
end

ds_plain = load_dataset("mnist", split="train")
@btime f(ds_plain)


ds_julia = with_format(ds_plain, "julia")
@btime f(ds_plain)


ds_py2jl = with_jltransform(ds_plain, py2jl)

set_transform!(ds2, py2jl)

@btime f1(ds2)

ds2[1]["image"]

set_transform!(ds2, identity)
@time ds2[1:10000]["image"];
@time Flux.batch(ds2[1:10000]["image"])

@time Flux.batch(ds2[1:10000]["image"] |> py2jl)

ds[1]

ds2["label"]

ds2.set_format("numpy")
ds2[1:10] |> py2jl

ds2["image"]

#####
function set_jltransform!(ds, transform = identity)
    ds.pyset_format("numpy")
    ds.jltransform(transform)
end
