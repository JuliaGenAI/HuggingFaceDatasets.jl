using Random, Statistics
using Flux
using Flux.Losses: logitcrossentropy
using Flux: onecold, onehotbatch
using HuggingFaceDatasets
using MLUtils: MLUtils, mapobs

# The image column is a stacked (W, H, N) UInt8 array, so just rescale to Float32 in [0, 1].
function mnist_transform(batch)
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
        ls += logitcrossentropy(ŷ, yoh, agg = sum)
        acc += sum(onecold(ŷ, 0:9) .== y)
        num += length(y)
    end
    return ls / num, acc / num
end

# `num_workers = 0` loads on the main process; `num_workers > 0` spreads each batch's `getobs`
# (and the CPython read it triggers) over that many worker processes, sidestepping the GIL — and as
# of MLUtils 0.4.12 the collated batch returns to the main process through **shared memory** (only a
# handle crosses the socket, not the pixels). `parallel=true` instead uses background threads. This
# is a tiny MLP, though, so the per-batch worker/thread overhead tends to outweigh any parallel-load
# gain — see the README.
#
# Returns the wall-clock seconds of `epochs` timed epochs. One extra warm-up epoch runs first and is
# discarded, so Julia's first-call JIT, worker-process startup, and the shm-session build stay out of
# the numbers — we measure steady-state per-epoch cost. (The DEMO, `verbose=true`, skips this and
# just reports accuracy per epoch.)
function train(; epochs = 4, num_workers = 0, materialize = false, parallel = false, verbose = true,
               loader_only = false)
    batchsize = 128
    nhidden = 100
    device = cpu

    train_data = load_dataset("ylecun/mnist", split = "train")
    test_data  = load_dataset("ylecun/mnist", split = "test")
    # Apply the transform lazily so it runs per batch during iteration (on the workers when
    # `num_workers > 0`); `mapobs`/`ObsView`-wrapped datasets compose with `num_workers`.
    train_data = mapobs(mnist_transform, train_data)
    test_data  = mapobs(mnist_transform, test_data)
    if materialize
        train_data = train_data[:]
        test_data  = test_data[:]
    end

    train_loader = Flux.DataLoader(train_data; batchsize, shuffle = true, num_workers, parallel)
    test_loader  = Flux.DataLoader(test_data; batchsize, num_workers, parallel)

    if loader_only
        # Iterate the training loader, consuming each batch but running no model — isolates data
        # loading (read + transform + collate + worker IPC) from compute. One warm-up epoch, then
        # time `epochs` epochs.
        for (x, y) in train_loader; end                       # warm-up (discarded)
        seen = 0
        return @elapsed for _ in 1:epochs
            for (x, y) in train_loader
                seen += length(y)
            end
        end
    end

    model = Chain(Flux.flatten,
                  Dense(28 * 28, nhidden, relu),
                  Dense(nhidden, nhidden, relu),
                  Dense(nhidden, 10)) |> device
    opt = Flux.setup(AdamW(1e-3), model)

    function train_epoch!()
        for (x, y) in train_loader
            x = x |> device
            yoh = onehotbatch(y, 0:9) |> device
            loss, grads = Flux.withgradient(m -> logitcrossentropy(m(x), yoh), model)
            Flux.update!(opt, model, grads[1])
        end
    end

    if verbose
        # DEMO: report train/test accuracy per epoch from initialization (no timing, no warm-up).
        function report(epoch)
            train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
            test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
            r(x) = round(x, digits = 3)
            r(x::Int) = x
            @info map(r, (; epoch, train_loss, train_acc, test_loss, test_acc))
        end
        report(0)
        for epoch in 1:epochs
            train_epoch!()
            report(epoch)
        end
        return
    end

    train_epoch!()                                            # warm-up (discarded)
    return @elapsed for _ in 1:epochs
        train_epoch!()
    end
end

const EPOCHS = parse(Int, get(ENV, "EPOCHS", "4"))

# Print the config name (flushed) before running it and the time after, so progress is visible live
# even when stdout is redirected to a file — Julia block-buffers a non-TTY stdout otherwise.
function timed(name; kwargs...)
    print("### ", rpad(name, 26)); flush(stdout)
    t = train(; epochs = EPOCHS, verbose = false, kwargs...)
    println(round(t; digits = 1), " seconds  ($EPOCHS epochs, warm-up discarded)"); flush(stdout)
end

println("### DEMO — MLP learning MNIST on CPU (accuracy per epoch)"); flush(stdout)
train(; epochs = EPOCHS, materialize = true, verbose = true)

# Each config below runs one warm-up epoch (discarded) before its timed epochs, so first-call JIT,
# worker-process startup, and the shm-session build never land in the reported time.
println("\n#### FULL TRAINING — model + data loading ###########"); flush(stdout)
MLUtils.close_dataloader_pool()
timed("Serial";                  num_workers = 0, materialize = false)
timed("Serial Materialized";     num_workers = 0, materialize = true)
timed("Parallel Materialized";   num_workers = 0, materialize = true, parallel = true)
timed("Distributed (2 workers)"; num_workers = 2, materialize = false)
timed("Distributed (4 workers)"; num_workers = 4, materialize = false)
timed("Distributed (8 workers)"; num_workers = 8, materialize = false)
MLUtils.close_dataloader_pool()
println("#### END FULL TRAINING ###########"); flush(stdout)

# Same configs, but iterating the loader with no model — the pure data-loading cost. Comparing
# against the full-training numbers shows how much of each config is loading vs. compute.
println("\n#### DATA-LOADING ONLY — no model, same $EPOCHS epochs ###########"); flush(stdout)
timed("Serial";                  num_workers = 0, materialize = false, loader_only = true)
timed("Serial Materialized";     num_workers = 0, materialize = true, loader_only = true)
timed("Parallel Materialized";   num_workers = 0, materialize = true, parallel = true, loader_only = true)
timed("Distributed (2 workers)"; num_workers = 2, materialize = false, loader_only = true)
timed("Distributed (4 workers)"; num_workers = 4, materialize = false, loader_only = true)
timed("Distributed (8 workers)"; num_workers = 8, materialize = false, loader_only = true)
MLUtils.close_dataloader_pool()
println("#### END DATA-LOADING ONLY ###########"); flush(stdout)
