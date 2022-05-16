using Test
using HuggingFaceDatasets, PythonCall, MLUtils
# using MLDatasets
# using ImageShow, ImageInTerminal

@testset "dataset" begin
    include("datasets.jl")
end
