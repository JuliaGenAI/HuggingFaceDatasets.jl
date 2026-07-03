using HuggingFaceDatasets
using Documenter

# Bring the public-but-unexported schema conveniences into doctest scope, and disable
# `datasets`' tqdm progress bars so doctests that trigger them (e.g. `class_encode_column`)
# don't emit a progress line into the captured output.
DocMeta.setdocmeta!(HuggingFaceDatasets, :DocTestSetup,
    :(using HuggingFaceDatasets, PythonCall;
      using HuggingFaceDatasets: features, class_names, int2str, str2int, Value;
      HuggingFaceDatasets.datasets.disable_progress_bars()); recursive=true)

makedocs(;
    modules=[HuggingFaceDatasets],
    authors="Carlo Lucibello <carlo.lucibello@gmail.com> and contributors",
    repo="https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/blob/{commit}{path}#{line}",
    sitename="HuggingFaceDatasets.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGenAI.github.io/HuggingFaceDatasets.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Guide" => "guide.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGenAI/HuggingFaceDatasets.jl",
    devbranch="main",
)
