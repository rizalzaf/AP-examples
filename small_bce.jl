using AdversarialPrediction
import AdversarialPrediction: define, constraint

using StatsBase
using DelimitedFiles
using Base.Iterators: partition

using LinearAlgebra

using Flux
using Flux.Tracker
using Flux: σ, logσ

using Printf, BSON
using Logging, LoggingExtras

using Random
Random.seed!(0)

include("common_metrics.jl")
include("pr@rc.jl")

# compute Tracker's gradient and get the objective
function gradient_obj(f, xs::Params)
    l = f()
    Tracker.losscheck(l)
    Tracker.@interrupts back!(l)
    gs = Tracker.Grads()
    for x in xs
        gs[Tracker.tracker(x)] = Tracker.extract_grad!(x)
    end
    return gs, l
end


function load_data(dname, id_split=1)

    @info "Loading: " * dname * ", split id = " * string(id_split)

    D_all = readdlm("data-cv/" * dname * ".csv", ',')
    id_train = readdlm("data-cv/" * dname * ".train", ',')
    id_test = readdlm("data-cv/" * dname * ".test", ',')

    id_train = round.(Int64, id_train)
    id_test = round.(Int64, id_test)

    ### Cross test, using first split
    ## First stage

    id_tr = vec(id_train[id_split,:])
    id_ts = vec(id_test[id_split,:])
    X_train = D_all[id_tr,1:end-1]
    y_train = round.(Int, D_all[id_tr, end])

    X_test = D_all[id_ts,1:end-1]
    y_test = round.(Int, D_all[id_ts, end])

    # transpose it, to align with sample-wise layout
    X_train = copy(X_train')
    X_test = copy(X_test')

    # Standardize into zero mean, unit variance
    dtrans = StatsBase.fit(StatsBase.ZScoreTransform, X_train)

    X_train = StatsBase.transform(dtrans, X_train)
    X_test = StatsBase.transform(dtrans, X_test)

    return X_train, y_train, X_test, y_test

end

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
    X_batch = Array{Float32}(undef, size(X, 1), length(idxs))
    for i in 1:length(idxs)
        X_batch[:, i] = Float32.(X[:,idxs[i]])
    end
    Y_batch = Y[idxs]
    return (X_batch, Y_batch)
end

function partition_minibatch(X, Y, batch_size)
    n = length(Y)
    mb_idxs = partition(1:n, batch_size)
    batch_set = [make_minibatch(X, Y, id) for id in mb_idxs]
end

# evaluate and accuracy
accuracy(x, y, model) = mean((model(x) .>= 0.0f0) .== y)
evaluate(x, y, model, pm::AdversarialPrediction.PerformanceMetric) = compute_metric(pm, Int.(Tracker.data(model(x) .>= 0.0f0)), y)
evaluate(x, y, model, pm::PrecisionGvRecall) = prec_at_rec(Tracker.data(model(x)), y, pm.th)


function run(dname::String, lambda::Real = 0.0, positive_class = 7:10)

    if lambda == 0.0
        attr = ""
    else
        attr = "-lambda-" * string(lambda)
    end

    # random seed
    Random.seed!(0)

    X_train, y_train, X_test, y_test = load_data(dname)

   
    # binary class
    y_train = binarize(y_train, positive_class)
    y_test = binarize(y_test, positive_class)
    

    # split training to train and validation
    n_tr = Int.(round(0.8 * length(y_train)))
    idr = randperm(length(y_train))

    # tr and val
    X_tr = X_train[:, idr[1:n_tr]]
    y_tr = y_train[idr[1:n_tr]]
    X_val = X_train[:, idr[n_tr+1:end]]
    y_val = y_train[idr[n_tr+1:end]]


    # minibatches
    batch_size = 25

    # minibatch
    train_set = partition_minibatch(X_tr, y_tr, batch_size)
    validation_set = (X_val, y_val)
    test_set = (X_test, y_test)

    # model
    nvar = size(X_train, 1)
    model = Chain(
        Dense(nvar, 100, relu),
        Dense(100, 100, relu),
        Dense(100, 1), vec) 
    

    # metric
    eval_metrics = [accuracy_metric, f1_score, f2_score, gpr, mcc, kappa, precision_gv_recall_80, precision_gv_recall_60]
    n_metric = length(eval_metrics)

    # optimizer
    eta = 1e-2
    n_epoch = 100

    opt = Descent(eta)


    # save results * objective
    n_batch = length(train_set)
    result_tr = zeros(n_metric, n_epoch + 1)
    result_val = zeros(n_metric, n_epoch + 1)
    obj_tr = zeros(n_batch, n_epoch)

    # logging save to file 
    log_filename = "log/BCE" * "-" * dname * attr * ".log"
    fc_logger = TeeLogger(FileLogger(log_filename), ConsoleLogger())
    global_logger(fc_logger)


    println()
    @info(@sprintf("λ: %.3f. Beginning training loop...", lambda))

    # training objective
    logitbinarycrossentropy(logŷ, y) = (1 .- y).*logŷ .- logσ.(logŷ)
    objective(x, y) = mean(logitbinarycrossentropy(model(x), y)) + lambda * sum(x -> sum(x .^ 2), params(model))  # l2 reg


    # metrics
    metrics_val = map(pm -> evaluate(validation_set..., model, pm), eval_metrics) 
    metrics_tr = map(pm -> evaluate(X_tr, y_tr, model, pm), eval_metrics)
    @info "0" * " | TR" * string(metrics_tr .|> Float64 .|> x -> round(x, digits=4)) *
    " VAL" * string(metrics_val .|> Float64 .|> x -> round(x, digits=4))

    result_val[:, 1] = metrics_val
    result_tr[:, 1] = metrics_tr


    best_metrics = -Inf * ones(n_metric)
    best_models = Vector(undef, n_metric) 
    for epoch_idx in 1:n_epoch

        # shuffle id
        sh_id = randperm(length(train_set))

        # Train for a single epoch
        # Flux.train!(objective, params(model), train_set[sh_id], opt)
        par = Flux.params(model)
        for ib = 1:n_batch
            dt = train_set[sh_id[ib]]

            # take gradients
            gs, obj = gradient_obj(par) do
                objective(dt...)
            end

            obj_tr[ib, epoch_idx] = Tracker.data(obj)

            # update params
            Tracker.update!(opt, par, gs)
        end


        # metrics
        metrics_val = map(pm -> evaluate(validation_set..., model, pm), eval_metrics) 
        metrics_tr = map(pm -> evaluate(X_tr, y_tr, model, pm), eval_metrics)
        @info string(epoch_idx) * " | TR" * string(metrics_tr .|> Float64 .|> x -> round(x, digits=4)) *
        " VAL" * string(metrics_val .|> Float64 .|> x -> round(x, digits=4))
        @show obj_tr[n_batch, epoch_idx]

        result_val[:, epoch_idx + 1] = metrics_val
        result_tr[:, epoch_idx + 1] = metrics_tr

        for it = 1:n_metric
            # If this is the best metric we've seen so far, save the model out
            if metrics_val[it] > best_metrics[it]
                @info("-> New best " * string(eval_metrics[it]) * "! : " * string(round(metrics_val[it], digits=4)))
                best_metrics[it] = metrics_val[it] 
                best_models[it] = model |> cpu
            end
        end

    end
   

    # model
    last_model = model
    metrics_val = map(pm -> evaluate(validation_set..., last_model, pm), eval_metrics) 

    println()
    @info("Training Finished")
    @info "BVAL" * string(best_metrics .|> Float64 .|> x -> round(x, digits=4)) *
        " LVAL" * string(metrics_val .|> Float64 .|> x -> round(x, digits=4))

    println()
    @info("Calculate final metrics...")

    result_dict = Dict{Symbol, Any}()
    result_dict[:lambda] = lambda
    result_dict[:best_metrics] = best_metrics
    result_dict[:best_models] = best_models
    result_dict[:result_tr] = result_tr
    result_dict[:result_val] = result_val
    result_dict[:obj_tr] = obj_tr

    return result_dict
end


function main(args)
    dname = args[1]

    ## load train test data, check n_class
    _, y_train, X_test, y_test = load_data(dname)
    n_class = maximum(y_train)
    
    # already binary
    if n_class == 1     # 0 and 1, already in binary formats
        pos_class = 1:1
    # dataset based
    elseif dname == "abalone"
        pos_class = 6:10
    elseif dname == "shuttle"
        pos_class = 4:7
    elseif dname == "letter"
        pos_class = 22:26
    elseif dname == "computeractivity2"
        pos_class = 8:10
    # n_class based
    elseif n_class == 10
        pos_class =  7:10
    elseif n_class == 7
        pos_class = 6:7
    elseif n_class == 5
        pos_class = 4:5
    else
        pos_class = 7:10
    end

    y_test = binarize(y_test, pos_class)
    test_set = (X_test, y_test)

    println("dname = ", dname, ", # class = ", n_class, ", positive class = ", pos_class)

    # lambdas
    lambdas = [0f0, 1f-3, 1f-2, 1f-1]
    nlambda = length(lambdas)

    all_results = Vector{Any}(undef, nlambda)

    for il = 1:length(lambdas)

        println()
        println("=========================")

        lambda = lambdas[il]
        dct = run(dname, lambda, pos_class)

        all_results[il] = dct
    end

    # test compute
    eval_metrics = [accuracy_metric, f1_score, f2_score, gpr, mcc, kappa, precision_gv_recall_80, precision_gv_recall_60]
    n_metric = length(eval_metrics)

    best_lambdas = zeros(n_metric)
    best_lambdas_id = zeros(Int, n_metric)
    best_lambdas_metric = -Inf .* ones(n_metric)
    for it = 1:n_metric
        for il = 1:nlambda
            if all_results[il][:best_metrics][it] > best_lambdas_metric[it]
                best_lambdas_metric[it] = all_results[il][:best_metrics][it]
                best_lambdas_id[it] = il
                best_lambdas[it] = lambdas[il]
            end
        end                
    end


    println()
    @info("Calculate final metrics...")
    println("=========================")


    metrics_ts_best = zeros(n_metric)
    for it = 1:n_metric
        best_model = all_results[best_lambdas_id[it]][:best_models][it]
        metrics_ts_best[it] = evaluate(test_set..., best_model, eval_metrics[it])
    end
    @info "Final | BTS" * string(metrics_ts_best .|> Float64 .|> x -> round(x, digits=4)) 

    # save best models
    model_filename = "BCE" * "-" * dname *  ".bson"
    fpath = joinpath(dirname(@__FILE__), "model", model_filename)
    BSON.@save fpath all_results best_lambdas best_lambdas_id best_lambdas_metric metrics_ts_best

    @info("DONE")

    return metrics_ts_best
end

# run main
metrics_ts_best = main(ARGS)