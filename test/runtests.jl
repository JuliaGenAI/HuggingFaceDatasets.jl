using Test
using HuggingFaceDatasets, PythonCall

PIL = HuggingFaceDatasets.PIL
np = HuggingFaceDatasets.np

@testset "dataset" begin
    include("dataset.jl")
end

@testset "datasetdict" begin
    include("datasetdict.jl")
end
