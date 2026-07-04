using Flux, Zygote
using Random, Statistics
using Flux.Losses: logitcrossentropy
using Flux: onecold, onehotbatch
using HuggingFaceDatasets
using MLUtils
# using ProfileView, BenchmarkTools

function mnist_transform(batch)
    # under the "julia" (= numpy) format the image column is already a stacked
    # (W, H, N) UInt8 array, so just rescale to Float32 in [0, 1]
    image = Float32.(batch["image"]) ./ 255f0
    return (; image, label = batch["label"])
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x = x |> device
        yoh = onehotbatch(y, 0:9) |> device
		ŷ = model(x)
        ls += logitcrossentropy(ŷ, yoh, agg=sum)
        acc += sum(onecold(ŷ, 0:9) .== y)
        num +=  length(y)
    end
    return ls / num, acc / num
end

# `num_workers = 0` loads on the main process; `num_workers > 0` spreads each batch's
# `getobs` (and the CPython read it triggers) over that many worker processes, sidestepping
# the GIL. MLUtils spawns the workers on demand under the current `--project`.
function train(; epochs=2, num_workers=0)
    batchsize = 128
    nhidden = 100
    device = cpu

    train_data = load_dataset("ylecun/mnist", split="train")
    test_data = load_dataset("ylecun/mnist", split="test")
    # apply the transform lazily so it runs per batch during iteration (on the workers when
    # `num_workers > 0`); `mapobs`/`ObsView`-wrapped datasets compose with `num_workers`
    train_data = mapobs(mnist_transform, train_data)
    test_data = mapobs(mnist_transform, test_data)

    train_loader = Flux.DataLoader(train_data; batchsize, shuffle=true, num_workers)
    test_loader = Flux.DataLoader(test_data; batchsize, num_workers)

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
	@time for epoch in 1:epochs
		for (x, y) in train_loader
			x = x |> device
			yoh = onehotbatch(y, 0:9) |> device
			loss, grads = withgradient(m -> logitcrossentropy(m(x), yoh), model)
            Flux.update!(opt, model, grads[1])
		end
        report(epoch)
	end
end

# @time train(; epochs=2, num_workers=0)
@time train(; epochs=2, num_workers=4)
