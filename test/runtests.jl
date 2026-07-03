using Test
using HuggingFaceDatasets, PythonCall, ImageCore

PIL = HuggingFaceDatasets.PIL
np = HuggingFaceDatasets.np

@testset "transforms" begin
    include("transforms.jl")
end

@testset "dataset" begin
    include("dataset.jl")
end

@testset "features" begin
    include("features.jl")
end

@testset "datasetdict" begin
    include("datasetdict.jl")
end

@testset "iterabledataset" begin
    include("iterabledataset.jl")
end

@testset "iterabledatasetdict" begin
    include("iterabledatasetdict.jl")
end

@testset "serialization" begin
    include("serialization.jl")
end

if !parse(Bool, get(ENV, "CI", "false"))
    @info "Testing larger datasets"
    @testset "larger datasets" begin
        include("no_ci.jl")
    end
else
    @info "CI detected: skipping tests on large datasets"
end
