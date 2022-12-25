using Test
using HuggingFaceDatasets, PythonCall, MLUtils
# using MLDatasets
# using ImageShow, ImageInTerminal

PIL = HuggingFaceDatasets.PIL
np = HuggingFaceDatasets.np

@testset "dataset" begin
    include("dataset.jl")
end

@testset "datasetdict" begin
    include("datasetdict.jl")
end
