# Julia data-access benchmarks for HuggingFaceDatasets.jl.
#
# Measures the data-loading / preprocessing operations that dominate a deep-learning
# input pipeline, on the MNIST test split (10k × 28×28 UInt8 images):
#
#   single   per-observation access:  getobs(ds, i)        for every i
#   batch    batched access:          getobs(ds, i:i+127)  over the whole split
#   full     materialize everything:  getobs(ds, :)
#   epoch    a realistic pass:         batch + cast to Float32/255 + one-hot labels
#
# across several formats, plus MLDatasets (native, fully in-memory) as a Julia baseline:
#
#   mldatasets   MLDatasets.MNIST                         (native Julia arrays)
#   plain        HF dataset, no format                    (raw Python objects / PIL images)
#   numpy        HF dataset, `set_format!(ds, "numpy")`   (raw NumPy, no py2jl)
#   julia        HF dataset, `"julia"` format             (numpy-backed + py2jl, the default)
#
# The Python counterpart (`perf.py`) runs the same tasks straight through `datasets`,
# so the two tables are directly comparable. See README.md for collected numbers.
#
# Run with this folder's project:
#   julia --project=perf perf/perf.jl

using HuggingFaceDatasets
using MLUtils: numobs, getobs
using MLDatasets: MNIST
using BenchmarkTools
using Printf
using Distributed

# --- tasks -----------------------------------------------------------------------------

single(ds) = for i in 1:numobs(ds)
    getobs(ds, i)
end

function batch(ds; bs = 128)
    n = numobs(ds)
    for i in 1:bs:n
        getobs(ds, i:min(i + bs - 1, n))
    end
end

full(ds) = getobs(ds, 1:numobs(ds))

# one-hot encode a vector of 0-based labels into a (10, N) Float32 matrix
function onehot10(labels)
    y = zeros(Float32, 10, length(labels))
    for (j, l) in enumerate(labels)
        y[l + 1, j] = 1f0
    end
    return y
end

# A realistic training-epoch pass over the data: read each batch, cast images to
# Float32 in [0, 1] and one-hot the labels, accumulating a checksum so nothing is
# optimized away. `image`/`label` accessors differ per source, hence the closures.
function epoch(ds, image, label; bs = 128)
    n = numobs(ds)
    s = 0.0f0
    for i in 1:bs:n
        b = getobs(ds, i:min(i + bs - 1, n))
        x = Float32.(image(b)) ./ 255f0
        y = onehot10(label(b))
        s += sum(x) + sum(y)
    end
    return s
end

# --- driver ----------------------------------------------------------------------------

# @belapsed in seconds -> ms
ms(f) = 1e3 * @belapsed $f()

function bench()
    mld    = MNIST(split = :test)
    base   = load_dataset("ylecun/mnist", split = "test")   # "julia" format by default
    plain  = set_format!(copy(base), nothing)               # raw Python objects / PIL images
    numpy  = with_format(base, "numpy")                     # raw NumPy, no py2jl
    julia  = with_format(base, "julia")                     # numpy-backed + py2jl (the default)

    # image/label accessors for the epoch task (batch layout differs per source)
    mld_img(b) = b.features;         mld_lab(b) = b.targets
    hf_img(b)  = b["image"];         hf_lab(b)  = b["label"]

    # The epoch/preprocessing task needs native Julia arrays; only the `julia` format
    # and MLDatasets hand those back (`plain`/`numpy` batches are raw Python objects,
    # so converting them *is* what the `julia` format does). Those get "—".
    variants = [
        ("mldatasets", mld,   mld_img, mld_lab),
        ("plain",      plain, nothing, nothing),
        ("numpy",      numpy, nothing, nothing),
        ("julia",      julia, hf_img,  hf_lab),
    ]

    rows = []
    for (name, ds, img, lab) in variants
        println(stderr, "benchmarking $name ...")
        t_single = ms(() -> single(ds))
        t_batch  = ms(() -> batch(ds))
        t_full   = ms(() -> full(ds))
        t_epoch  = img === nothing ? nothing : ms(() -> epoch(ds, img, lab))
        push!(rows, (name, t_single, t_batch, t_full, t_epoch))
    end

    cell(x) = x === nothing ? @sprintf("%10s", "—") : @sprintf("%10.1f", x)
    println("\n## Julia (HuggingFaceDatasets.jl) — MNIST test, times in ms\n")
    @printf("| %-11s | %10s | %10s | %10s | %10s |\n",
            "variant", "single", "batch", "full", "epoch")
    @printf("|%s|%s|%s|%s|%s|\n", "-"^13, "-"^12, "-"^12, "-"^12, "-"^12)
    for (name, s, b, f, e) in rows
        @printf("| %-11s | %s | %s | %s | %s |\n", name, cell(s), cell(b), cell(f), cell(e))
    end
end

# --- parallel loading across worker processes -------------------------------------------
#
# Thread-based `parallel=true` cannot speed up a PythonCall read: the CPython GIL serializes
# it, so N threads read no faster than one (and can segfault without GIL guards). Separate
# worker *processes* each have their own interpreter/GIL, so the reads run in parallel. A
# `Dataset` is serializable — it ships a by-reference pickle of its on-disk Arrow files and
# each worker re-mmaps them (no data copy) — so a process pool reads batches concurrently.
# This is the mechanism a process-based `DataLoader(ds; num_workers=N)` would build on. Here
# we read the whole split in batches, serially vs across N worker processes.
function bench_parallel(; bs = 128, worker_counts = (2, 4))
    base    = load_dataset("ylecun/mnist", split = "test")
    ds      = with_format(base, "julia")
    n       = numobs(ds)
    idxsets = [i:min(i + bs - 1, n) for i in 1:bs:n]
    readf   = ix -> getobs(ds, ix)   # a single closure ⇒ CachingPool ships `ds` once per worker

    println(stderr, "benchmarking parallel loading ...")
    t_serial = ms(() -> foreach(readf, idxsets))
    rows = [("serial", t_serial, 1.0)]

    procs = addprocs(maximum(worker_counts); exeflags = `--project=$(dirname(Base.active_project()))`)
    try
        @everywhere procs using HuggingFaceDatasets, MLUtils
        for k in worker_counts
            pool = CachingPool(procs[1:k])
            pmap(readf, pool, idxsets)                    # warm up: caches `ds` on the k workers
            t = ms(() -> pmap(readf, pool, idxsets))
            push!(rows, ("$k procs", t, t_serial / t))
        end
    finally
        rmprocs(procs)
    end

    println("\n## Parallel loading (process pool) — MNIST test, batchsize $bs, whole split\n")
    @printf("| %-9s | %10s | %8s |\n", "workers", "time (ms)", "speedup")
    @printf("|%s|%s|%s|\n", "-"^11, "-"^12, "-"^10)
    for (name, t, sp) in rows
        @printf("| %-9s | %10.1f | %7.2fx |\n", name, t, sp)
    end
end

bench()
bench_parallel()
