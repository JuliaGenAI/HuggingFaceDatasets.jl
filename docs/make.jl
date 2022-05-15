using HuggingFaceDatasets
using Documenter

DocMeta.setdocmeta!(HuggingFaceDatasets, :DocTestSetup, :(using HuggingFaceDatasets); recursive=true)

makedocs(;
    modules=[HuggingFaceDatasets],
    authors="Carlo Lucibello <carlo.lucibello@gmail.com> and contributors",
    repo="https://github.com/CarloLucibello/HuggingFaceDatasets.jl/blob/{commit}{path}#{line}",
    sitename="HuggingFaceDatasets.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://CarloLucibello.github.io/HuggingFaceDatasets.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/CarloLucibello/HuggingFaceDatasets.jl",
    devbranch="main",
)
