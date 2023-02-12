using Test
using HuggingFaceDatasets, PythonCall, ImageCore

PIL = HuggingFaceDatasets.PIL
np = HuggingFaceDatasets.np

@testset "dataset" begin
    include("dataset.jl")
end

@testset "datasetdict" begin
    include("datasetdict.jl")
end

if !parse(Bool, get(ENV, "CI", "false"))
    @info "Testing larger datasets"
    @testset "larger datasets" begin
        include("no_ci.jl")
    end
else
    @info "CI detected: skipping tests on large datasets"
end
