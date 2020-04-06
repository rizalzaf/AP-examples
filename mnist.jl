using AdversarialPrediction
import AdversarialPrediction: define, constraint

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, mapleaves
using Base.Iterators: repeated, partition
using Printf, BSON
using Logging, LoggingExtras

using LinearAlgebra

using CuArrays
using CUDAdrv
using CUDAnative

using Random
Random.seed!(0)

include("common_metrics.jl")
include("pr@rc.jl")


# make multiclass to binary problem
function binarize(y, positive_classes = [1])
    n = length(y)
    y_bin = zeros(Int, n)
    isin(x) = Bool(x in positive_classes)
    y_bin[isin.(y)] .= 1

    return y_bin
end

# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = Y[idxs]
    return (X_batch, Y_batch)
end

function compute_metric_batch(pm::PerformanceMetric, batched_set::AbstractArray, model, gc_iter = 100)

    n_batch = length(batched_set)
    vy = Vector{Vector{Float32}}(undef, n_batch)
    vps = Vector{Vector{Float32}}(undef, n_batch)
    
    for i = 1:n_batch
        vy[i] = cpu(batched_set[i][2])
        vps[i] = cpu(model(batched_set[i][1]))

        if i % gc_iter == 0
            GC.gc()     # garbage collector, reduce memmory req.
        end
    end
    y = Int.(vcat(vy...))
    yhat = Int.(vcat(vps...) .>= 0)

    return compute_metric(pm, yhat, y)
end

function compute_metric_batch(eval_metrics::Vector{<:PerformanceMetric}, batched_set::AbstractArray, model, gc_iter = 100)

    n_batch = length(batched_set)
    vy = Vector{Vector{Float32}}(undef, n_batch)
    vps = Vector{Vector{Float32}}(undef, n_batch)
    
    for i = 1:n_batch
        vy[i] = cpu(batched_set[i][2])
        vps[i] = cpu(model(batched_set[i][1]))

        if i % gc_iter == 0
            GC.gc()     # garbage collector, reduce memmory req.
        end
    end
    y = Int.(vcat(vy...))
    yhat = Int.(vcat(vps...) .>= 0)

    return map(pm -> compute_metric(pm, yhat, y), eval_metrics)
end

function compute_metric_batch(pm::PerformanceMetric, data_set::Tuple, model)

    y = cpu(data_set[2])
    yhat = Int.(cpu(model(data_set[1])) .>= 0)

    return compute_metric(pm, yhat, y)
end

function run(pm::PerformanceMetric, lambda::Real = 0.0)

    # random seed
    Random.seed!(0)

    # Load labels and images from Flux.Data.MNIST
    @info("Loading data set")
    train_labels = binarize(MNIST.labels(), 0:0)
    train_imgs = MNIST.images()


    # split training to train and validation
    n_tr = Int.(round(0.8 * length(train_labels)))
    idr = randperm(length(train_labels))

    tr_labels = train_labels[1:n_tr]
    tr_imgs = train_imgs[1:n_tr]

    val_labels = train_labels[n_tr+1:end]
    val_imgs = train_imgs[n_tr+1:end]

    # batching
    batch_size = 25
    mb_idxs = partition(1:length(tr_imgs), batch_size)
    train_set = [make_minibatch(tr_imgs, tr_labels, i) for i in mb_idxs]

    # validation set
    mb_idxs_val = partition(1:length(val_imgs), batch_size)
    # validation_set = make_minibatch(val_imgs, val_labels, 1:length(val_imgs))
    validation_set = [make_minibatch(val_imgs, val_labels, i) for i in mb_idxs_val]
    
    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_labels = binarize(MNIST.labels(:test), 0:0)

    mb_idxs_test = partition(1:length(test_imgs), batch_size)
    # test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))
    test_set = [make_minibatch(test_imgs, test_labels, i) for i in mb_idxs_test]


    @info("Constructing model...")
    cv2dense(x) = reshape(x, :, size(x, 4))

    model = Chain(
        Conv((5, 5), 1=>20, stride=(1,1), relu),
        MaxPool((2,2)),
        Conv((5, 5), 20=>50, stride=(1,1), relu),
        MaxPool((2,2)),
        cv2dense,
        Dense(4*4*50, 500),
        Dense(500, 1), vec
    )

    # model = Chain(
    #     Conv((5, 5), 1=>16, stride=(2,2), relu),
    #     MaxPool((2,2)),
    #     cv2dense,
    #     Dense(6*6*16, 1), vec
    # )

    # Load model and datasets onto GPU, if enabled
    train_set = gpu.(train_set)
    validation_set = gpu.(validation_set)
    test_set = gpu.(test_set)
    model = gpu(model)

    
    # metric computations
    accuracy(x, y, model) = mean((model(x) .>= 0.0f0) .== y)
    evaluate(x, y, model, pm::AdversarialPrediction.PerformanceMetric) = compute_metric(pm, Int.(model(x) .>= 0.0f0), y)
    evaluate(x, y, model, pm::PrecisionGvRecall) = prec_at_rec(model(x), y, pm.th)


    # optimizer
    eta = 0.001 
    n_epoch = 100
    admm_iter = 100
    
    opt = Descent(eta)

    # save results * objective
    n_batch = length(train_set)
    result_tr = zeros(n_epoch + 1)
    result_val = zeros(n_epoch + 1)

    # logging save to file 
    log_filename = "log/" * "AP-MNIST-" * string(pm) * ".log"
    fc_logger = TeeLogger(FileLogger(log_filename), ConsoleLogger())
    global_logger(fc_logger)    


    println()
    @info(@sprintf("λ: %.3f. Beginning training loop...", lambda))

    # training objective
    objective(x, y) = ap_objective(model(x), y, pm; max_iter = admm_iter) + lambda * sum(x -> sum(x .* x), params(model))  # l2 reg


    # val = evaluate(validation_set..., model, pm)
    val = compute_metric_batch(pm, validation_set, model)
    val_tr = compute_metric_batch(pm, train_set, model)
    @info(@sprintf("λ: %.3f. [0]: [%s] Train metric: %.5f, validation metric: %.5f", lambda, string(pm), val_tr, val))

    result_val[1] = val
    result_tr[1] = val_tr

    best_val = 0.0
    best_model = model |> cpu

    for epoch_idx in 1:n_epoch

        @time begin
        
        # shuffle id
        sh_id = randperm(n_batch)

        # Train for a single epoch
        Flux.train!(objective, params(model), train_set[sh_id], opt)

        # Calculate metric:
        # val = evaluate(validation_set..., model, pm)
        val = compute_metric_batch(pm, validation_set, model)
        val_tr = compute_metric_batch(pm, train_set, model)
        @info(@sprintf("λ: %.3f. [%d]: [%s] Train metric: %.5f, validation metric: %.5f", lambda, epoch_idx, string(pm), val_tr, val))

        result_val[epoch_idx + 1] = val
        result_tr[epoch_idx + 1] = val_tr

        # If this is the best accuracy we've seen so far, save the model out
        if val > best_val
            @info("-> New best validation metric! : " * string(round(val, digits=4)))
            best_val = val 
            best_model = model |> cpu      # make a copy
        end

        end

    end

    
   # model
    last_model = model
    val = compute_metric_batch(pm, validation_set, last_model)
    
    println()
    @info("Training Finished")
    @info(@sprintf("λ: %.3f. Validation metric -> best: %.5f, last: %.5f", lambda, best_val, val))


    result_dict = Dict{Symbol, Any}()
    result_dict[:lambda] = lambda
    result_dict[:best_val] = best_val
    result_dict[:best_model] = best_model
    result_dict[:result_tr] = result_tr
    result_dict[:result_val] = result_val

    return result_dict

end

function main(args)
    metric_str = args[1]

    # metric
    pm = f1_score

    if metric_str == "f2"
        pm = f2_score
    elseif metric_str == "gpr"
        pm = gpr
    elseif metric_str == "mcc"
        pm = mcc  #pm = inform
    elseif metric_str == "kappa"
        pm = kappa
    elseif metric_str == "pr8"
        pm = precision_gv_recall_80
    elseif metric_str == "pr6"
        pm = precision_gv_recall_60
    elseif metric_str == "acc"
        pm = accuracy_metric
    end

    if length(args) > 1
        gpu_id = parse(Int, args[2])
        CUDAnative.device!(gpu_id)
    end


    # lambdas
    lambdas = [0f0, 1f-3, 1f-2, 1f-1]
    nlambda = length(lambdas)

    all_results = Vector{Any}(undef, nlambda)

    for il = 1:length(lambdas)

        println()
        println("=========================")

        lambda = lambdas[il]
        dct = run(pm, lambda)

        all_results[il] = dct
    end

    best_lambda = 0.0
    best_lambda_id = 0
    best_lambda_val = -Inf 
    for il = 1:nlambda
        if all_results[il][:best_val] > best_lambda_val
            best_lambda_val = all_results[il][:best_val]
            best_lambda_id = il
            best_lambda = lambdas[il]
        end
    end                


    println()
    @info("Calculate final metrics...")
    println("=========================")

    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_labels = binarize(MNIST.labels(:test), 0:0)
    
    batch_size = 25
    mb_idxs_test = partition(1:length(test_imgs), batch_size)
    # test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))
    test_set = [make_minibatch(test_imgs, test_labels, i) for i in mb_idxs_test]

    test_set = gpu.(test_set)

    # Calculate metric:
    best_model = all_results[best_lambda_id][:best_model]
    val_ts_best = compute_metric_batch(pm, test_set, gpu(best_model))
    @info(@sprintf("Final : [%s] Test metric -> best model: %.5f", string(pm), val_ts_best))

    # save best models
    model_filename = "AP-" * "MNIST" * "-" * string(pm) * ".bson"
    fpath = joinpath(dirname(@__FILE__), "model", model_filename)
    BSON.@save fpath all_results best_lambda best_lambda_id best_lambda_val val_ts_best

    @info("DONE")

    return val_ts_best
end

# run main
val_ts_best = main(ARGS)