
using Flux

function cifar_transform(x)
    x = py2jl(x)
    image = Flux.batch(x["img"]) ./ 255f0
    label = Flux.onehotbatch(x["label"], 0:9)
    return (; image, label)
end

dtrain = load_dataset("cifar10", split="train")
set_transform!(dtrain, cifar_transform)
