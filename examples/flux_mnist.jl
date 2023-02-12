using Flux, Zygote
using Random, Statistics
using Flux.Losses: logitcrossentropy
using Flux: onecold
using HuggingFaceDatasets
using MLUtils
using ImageCore
# using ProfileView, BenchmarkTools

function mnist_transform(batch)
    image = ImageCore.channelview.(batch["image"]) # from Matrix{Gray{N0f8}} to Matrix{UInt8}
    image = Flux.batch(image) ./ 255f0
    label = Flux.onehotbatch(batch["label"], 0:9)
    return (; image, label)
end

# Remove when https://github.com/JuliaML/MLUtils.jl/pull/147 is merged and tagged
Base.getindex(data::MLUtils.MappedData, idx::Int) = getobs(data.f(getobs(data.data, [idx])), 1)
Base.getindex(data::MLUtils.MappedData, idxs::AbstractVector) = data.f(getobs(data.data, idxs))
Base.getindex(data::MLUtils.MappedData, ::Colon) = data[1:length(data.data)]


function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = x |> device, y |> device
		ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y))
        num +=  size(x)[end]
    end
    return ls / num, acc / num
end

function train(epochs)
    batchsize = 128
    nhidden = 100
    device = cpu

    train_data = load_dataset("mnist", split="train").with_format("julia")
    test_data = load_dataset("mnist", split="test").with_format("julia")
    train_data = mapobs(mnist_transform, train_data)[:] # lazy apply transform then materialize
    test_data = mapobs(mnist_transform, test_data)[:]
    
    train_loader = Flux.DataLoader(train_data; batchsize, shuffle=true) 
    test_loader = Flux.DataLoader(test_data; batchsize)
    
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
			x, y = x |> device, y |> device
			loss, grads = withgradient(model -> logitcrossentropy(model(x), y), model)
            Flux.update!(opt, model, grads[1])
		end
        report(epoch)
	end
end

@time train(2)  # 8s on a m1 pro with in-memory loading
                # 20s on-the-fly loading
