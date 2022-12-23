using Test
using HuggingFaceDatasets, PythonCall, MLUtils
# using MLDatasets
# using ImageShow, ImageInTerminal


@testset "load_dataset" begin
    include("load_dataset.jl")
end

@testset "dataset" begin
    include("dataset.jl")
end

@testset "datasetdict" begin
    include("datasetdict.jl")
end
