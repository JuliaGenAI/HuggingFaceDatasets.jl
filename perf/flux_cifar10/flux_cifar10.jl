using Random, Statistics
using Flux
using Flux.Losses: logitcrossentropy
using Flux: onecold, onehotbatch
using HuggingFaceDatasets
using MLUtils: MLUtils, mapobs
using CUDA, cuDNN

# Train on the GPU when one is available (this example is written for it), else fall back to CPU.
const DEVICE = CUDA.functional() ? gpu : cpu

# CIFAR-10 per-channel mean/std (the standard values), shaped for (W, H, C, N) broadcasting.
const CIFAR_MEAN = reshape(Float32[0.4914, 0.4822, 0.4465], 1, 1, 3, 1)
const CIFAR_STD  = reshape(Float32[0.2470, 0.2435, 0.2616], 1, 1, 3, 1)

# Decode a raw batch to normalized WHCN Float32. Under the "julia" format the image column is a
# stacked (C, W, H, N) UInt8 array — channel axis first, from the numpy→Julia axis reversal — so
# permute to Flux's (W, H, C, N) and standardize per channel. Deterministic, hence materializable.
function cifar_decode(batch)
    x = Float32.(batch["img"]) ./ 255f0       # (C, W, H, N)
    x = permutedims(x, (2, 3, 1, 4))          # (W, H, C, N) — Flux WHCN layout
    x = (x .- CIFAR_MEAN) ./ CIFAR_STD
    return (; image = x, label = batch["label"])
end

# Standard CIFAR-10 training augmentation, applied per batch: zero-pad by 4 and take a random
# 32×32 crop, then flip horizontally with probability 1/2 — the classic crop+flip pipeline. Pure
# Julia (no Python), so it parallelizes across threads (`parallel=true`) and worker processes
# (`num_workers`). It is random per call, so it must run every epoch and must not be materialized.
function cifar_augment(batch)
    x = batch.image
    W, H, C, N = size(x)
    pad = 4
    out = similar(x)
    padded = zeros(Float32, W + 2pad, H + 2pad, C)
    for n in 1:N
        fill!(padded, 0f0)
        @views padded[pad+1:pad+W, pad+1:pad+H, :] .= x[:, :, :, n]
        i, j = rand(0:2pad), rand(0:2pad)                  # random top-left crop offset
        if rand(Bool)
            @views out[:, :, :, n] .= padded[i+W:-1:i+1, j+1:j+H, :]   # crop + horizontal flip
        else
            @views out[:, :, :, n] .= padded[i+1:i+W, j+1:j+H, :]      # crop
        end
    end
    return (; image = out, label = batch.label)
end

# On-the-fly training pipeline: decode then augment. A named function (not a closure) so the
# `num_workers` path can ship it — and the module globals it reads — to worker processes.
cifar_train_transform(batch) = cifar_augment(cifar_decode(batch))

# A small VGG-style CNN: three (conv-conv-pool) blocks of increasing width, then a classifier.
# BatchNorm speeds up convergence; both it and Dropout switch behavior between train/eval, which
# Flux handles automatically (active inside `withgradient`, inactive during plain inference).
function make_model()
    return Chain(
        Conv((3, 3), 3 => 64, pad = 1, bias = false),   BatchNorm(64, relu),
        Conv((3, 3), 64 => 64, pad = 1, bias = false),  BatchNorm(64, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, pad = 1, bias = false),  BatchNorm(128, relu),
        Conv((3, 3), 128 => 128, pad = 1, bias = false), BatchNorm(128, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 128 => 256, pad = 1, bias = false), BatchNorm(256, relu),
        Conv((3, 3), 256 => 256, pad = 1, bias = false), BatchNorm(256, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(256 * 4 * 4, 256, relu),
        Dropout(0.5),
        Dense(256, 10),
    )
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x = x |> device
        ŷ = model(x) |> cpu                  # logits back on the CPU for cheap metric bookkeeping
        yoh = onehotbatch(y, 0:9)
        ls += logitcrossentropy(ŷ, yoh, agg = sum)
        acc += sum(onecold(ŷ, 0:9) .== y)
        num += length(y)
    end
    return ls / num, acc / num
end

# `num_workers = 0` loads on the main process; `num_workers > 0` spreads each batch's `getobs`
# (the CPython image decode) over that many worker processes, sidestepping the GIL — and as of
# MLUtils 0.4.12 the collated batch returns to the main process through **shared memory** (only a
# handle crosses the socket, not the ~1.5 MB of pixels), so the process-parallel path now actually
# scales. `parallel=true` instead uses background threads — useful once the data is materialized,
# where the per-batch work left is the pure-Julia augmentation (no GIL to serialize it).
#
# Returns the wall-clock seconds of `epochs` timed epochs. One extra warm-up epoch runs first and is
# discarded, so Julia's first-call JIT, worker-process startup, and the shm-session build stay out of
# the numbers — we measure steady-state per-epoch cost. (The DEMO, `verbose=true`, skips this and
# just reports accuracy per epoch from initialization.)
function train(; epochs = 5, num_workers = 0, materialize = false, parallel = false, verbose = true,
               loader_only = false)
    batchsize = 128
    device = DEVICE

    train_ds = load_dataset("uoft-cs/cifar10", split = "train")
    test_ds  = load_dataset("uoft-cs/cifar10", split = "test")

    if materialize
        # Decode once into memory, then augment per batch on top of the in-memory tensors: the
        # per-epoch Python decode is gone and only the (pure-Julia) augmentation remains.
        train_base = mapobs(cifar_decode, train_ds)[:]
        train_data = mapobs(cifar_augment, train_base)
        test_data  = mapobs(cifar_decode, test_ds)[:]
    else
        # Decode + augment on the fly every batch; the CPython decode runs under the GIL.
        train_data = mapobs(cifar_train_transform, train_ds)
        test_data  = mapobs(cifar_decode, test_ds)
    end

    train_loader = Flux.DataLoader(train_data; batchsize, shuffle = true, num_workers, parallel)
    test_loader  = Flux.DataLoader(test_data; batchsize, num_workers, parallel)

    if loader_only
        # Iterate the training loader, consuming each batch but running no model — isolates data
        # loading (decode + augment + collate + worker IPC) from GPU compute. One warm-up epoch is
        # discarded, then `epochs` epochs are timed.
        for (x, y) in train_loader; end                       # warm-up (discarded)
        seen = 0
        return @elapsed for _ in 1:epochs
            for (x, y) in train_loader
                seen += length(y)
            end
        end
    end

    model = make_model() |> device
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

const EPOCHS = parse(Int, get(ENV, "EPOCHS", "10"))

report_time(t) = println("  ", round(t; digits = 1), " seconds  ($EPOCHS epochs, warm-up discarded)")

println("### DEMO — CNN learning CIFAR-10 on ", DEVICE === gpu ? "GPU" : "CPU", " (accuracy per epoch)")
train(; epochs = EPOCHS, materialize = true, verbose = true)

# Each config below runs one warm-up epoch (discarded) before its timed epochs, so first-call JIT,
# worker-process startup, and the shm-session build never land in the reported time.
println("\n#### FULL TRAINING — model + data loading ###########")
MLUtils.close_dataloader_pool()
println("### Serial");                  report_time(train(; epochs = EPOCHS, num_workers = 0, materialize = false, verbose = false))
println("### Serial Materialized");     report_time(train(; epochs = EPOCHS, num_workers = 0, materialize = true,  verbose = false))
println("### Parallel Materialized");   report_time(train(; epochs = EPOCHS, num_workers = 0, materialize = true, parallel = true, verbose = false))
println("### Distributed (2 workers)"); report_time(train(; epochs = EPOCHS, num_workers = 2, materialize = false, verbose = false))
println("### Distributed (4 workers)"); report_time(train(; epochs = EPOCHS, num_workers = 4, materialize = false, verbose = false))
println("### Distributed (8 workers)"); report_time(train(; epochs = EPOCHS, num_workers = 8, materialize = false, verbose = false))
MLUtils.close_dataloader_pool()
println("#### END FULL TRAINING ###########")

# Same configs, but iterating the loader with no model — the pure data-loading cost. Comparing
# against the full-training numbers shows how much of each config is loading vs. GPU compute.
println("\n#### DATA-LOADING ONLY — no model, same $EPOCHS epochs ###########")
println("### Serial");                  report_time(train(; epochs = EPOCHS, num_workers = 0, materialize = false, verbose = false, loader_only = true))
println("### Serial Materialized");     report_time(train(; epochs = EPOCHS, num_workers = 0, materialize = true,  verbose = false, loader_only = true))
println("### Parallel Materialized");   report_time(train(; epochs = EPOCHS, num_workers = 0, materialize = true, parallel = true, verbose = false, loader_only = true))
println("### Distributed (2 workers)"); report_time(train(; epochs = EPOCHS, num_workers = 2, materialize = false, verbose = false, loader_only = true))
println("### Distributed (4 workers)"); report_time(train(; epochs = EPOCHS, num_workers = 4, materialize = false, verbose = false, loader_only = true))
println("### Distributed (8 workers)"); report_time(train(; epochs = EPOCHS, num_workers = 8, materialize = false, verbose = false, loader_only = true))
MLUtils.close_dataloader_pool()
println("#### END DATA-LOADING ONLY ###########")
