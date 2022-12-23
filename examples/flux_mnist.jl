using Flux, Zygote
using Random, Statistics
using Flux.Losses: logitcrossentropy
using Flux: onecold
using HuggingFaceDatasets
# using ProfileView, BenchmarkTools

function mnist_transform(x)
    x = py2jl(x)
    image = Flux.batch(x["image"]) ./ 255f0
    label = Flux.onehotbatch(x["label"], 0:9)
    return (; image, label)
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
        num +=  size(x)[end]
    end
    return ls / num, acc / num
end

function train(epochs)	
    batchsize = 128
    nhidden = 100
    device = gpu

    dataset = load_dataset("mnist")
    set_transform!(dataset, mnist_transform)

    # We use [:] to materialize and transform the whole dataset.
    # This gives much faster iterations.
    train_loader = Flux.DataLoader(dataset["train"][1:1000]; batchsize, shuffle=true) 
    test_loader = Flux.DataLoader(dataset["test"][1:1000]; batchsize)

    model = Chain([Flux.flatten,
                   Dense(28*28, nhidden, relu),
                   Dense(nhidden, nhidden, relu),
                   Dense(nhidden, 10)]) |> device

	opt = Flux.setup(AdamW(1e-3), model)

    function report(epoch)
		train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
		test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
        r(x) = round(x, digits=3) 
		r(x::Int) = x 
        @info map(r, (; epoch, train_loss, train_acc, test_loss, test_acc))
    end

    report(0)
	for epoch in 1:epochs
		for (x, y) in train_loader
			x, y = x |> device, y |> device
			loss, grads = withgradient(model -> logitcrossentropy(model(x), y), model)
            Flux.update!(opt, model, grads[1])
		end
        report(epoch)
	end
end

@time train(2)
