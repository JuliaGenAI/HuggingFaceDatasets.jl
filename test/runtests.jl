using Test
using HuggingFaceDatasets, PythonCall, MLUtils
# using MLDatasets
# using ImageShow, ImageInTerminal

@testset "dataset" begin
    include("dataset.jl")
end

@testset "datasetdict" begin
    include("datasetdict.jl")
end
