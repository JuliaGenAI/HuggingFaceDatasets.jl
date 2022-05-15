using Flux, Zygote
using Random, Statistics
using Flux.Losses: logitcrossentropy
using Flux: onecold
using CUDA
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
    device = cpu

    dtrain = load_dataset("mnist", split="train")
    dtest = load_dataset("mnist", split="test")
    set_transform!(dtrain, mnist_transform)
    set_transform!(dtest, mnist_transform)

    # We use [:] to materialize and transform the whole dataset.
    # This gives much faster iterations.
    train_loader = Flux.DataLoader(dtrain[:]; batchsize, shuffle=true) 
    test_loader = Flux.DataLoader(dtest[:]; batchsize)

    model = Chain([Flux.flatten,
                   Dense(28*28, nhidden, relu),
                   Dense(nhidden, nhidden, relu),
                   Dense(nhidden, 10)]) |> device

	ps = Flux.params(model)
	opt = ADAM(1e-4)

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
			loss, gs = withgradient(() -> logitcrossentropy(model(x), y), ps)
			Flux.update!(opt, ps, gs)
		end
        report(epoch)
	end
end

@time train(2)
