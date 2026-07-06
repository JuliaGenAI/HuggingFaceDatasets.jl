using Random, Statistics
using Flux
using Flux.Losses: logitcrossentropy
using Flux: onecold, onehotbatch
using HuggingFaceDatasets
using MLUtils: MLUtils, mapobs
# using ProfileView, BenchmarkTools

function mnist_transform(batch)
    # the image column is a stacked (W, H, N) UInt8 array,
    # so just rescale to Float32 in [0, 1]
    image = batch["image"] ./ 255f0
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
function train(; epochs=2, num_workers=0, materialize=false, parallel=false, verbose=true)
    batchsize = 128
    nhidden = 100
    device = cpu

    train_data = load_dataset("ylecun/mnist", split="train")
    test_data = load_dataset("ylecun/mnist", split="test")
    # apply the transform lazily so it runs per batch during iteration (on the workers when
    # `num_workers > 0`); `mapobs`/`ObsView`-wrapped datasets compose with `num_workers`
    train_data = mapobs(mnist_transform, train_data)
    test_data = mapobs(mnist_transform, test_data)
    if materialize
        train_data = train_data[:]
        test_data = test_data[:]
    end

    train_loader = Flux.DataLoader(train_data; batchsize, shuffle=true, num_workers, parallel)
    test_loader = Flux.DataLoader(test_data; batchsize, num_workers, parallel)

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

    verbose && report(0)
	for epoch in 1:epochs
		for (x, y) in train_loader
			x = x |> device
			yoh = onehotbatch(y, 0:9) |> device
			loss, grads = Flux.withgradient(m -> logitcrossentropy(m(x), yoh), model)
            Flux.update!(opt, model, grads[1])
		end
        verbose && report(epoch)
	end
end

println("#### START COMPARISON ###########")
MLUtils.close_dataloader_pool()
println("### WARMUP") # for precompilation
@time train(; epochs=1, num_workers=0, materialize=false, verbose=false)
println("### Serial")
@time train(; epochs=4, num_workers=0, materialize=false, verbose=false)
println("### Serial Materialized")
@time train(; epochs=4, num_workers=0, materialize=true, verbose=false)
println("### Paralle Materialized")
@time train(; epochs=4, num_workers=0, materialize=true, parallel=true, verbose=false)
println("### Distributed")
@time train(; epochs=4, num_workers=4, materialize=false, verbose=false)
MLUtils.close_dataloader_pool()
println("#### END COMPARISON ###########")
