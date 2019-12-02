using StatsBase
using DelimitedFiles
using Base.Iterators: partition

using LinearAlgebra

using Flux
using Flux: σ, logσ
using CuArrays
using CUDAdrv

using CUDAnative
CUDAnative.device!(0)

using Printf, BSON

using Random
Random.seed!(0)

include("common_metrics.jl")

function load_data(dname, id_split=1)

    @info "Loading: " * dname * ", split id = " * string(id_split)

    D_all = readdlm("data-cv/" * dname * ".csv", ',')
    id_train = readdlm("data-cv/" * dname * ".train", ',')
    id_test = readdlm("data-cv/" * dname * ".test", ',')

    id_train = round.(Int64, id_train)
    id_test = round.(Int64, id_test)

    ### Cross Validation, using first split
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


dname = "whitewine"
X_train, y_train, X_test, y_test = load_data(dname)

# binary class
y_train = binarize(y_train, 7:10)
y_test = binarize(y_test, 7:10)


# split training to train and validation
n_tr = Int.(round(0.7 * length(y_train)))
idr = randperm(length(y_train))

# tr and val
X_tr = X_train[:, 1:n_tr]
y_tr = y_train[1:n_tr]
X_val = X_train[:, n_tr+1:end]
y_val = y_train[n_tr+1:end]


# minibatches
batch_size = 25

# minibatch
train_set = gpu.(partition_minibatch(X_tr, y_tr, batch_size))
validation_set = gpu.((X_val, y_val))
test_set = gpu.((X_test, y_test))


# model
nvar = size(X_tr, 1)
model = Chain(
    Dense(nvar, 50, relu),
    Dense(50, 50, relu),
    Dense(50, 1), vec) |> gpu


# metric
pm = f1_score
# pm = f2_score
# pm = gpr
# pm = inform
# pm = kappa
# pm = precision_gv_recall_80
# pm = precision_gv_recall_60
# pm = accuracy_metric

# using ECOS
# solver = with_optimizer(ECOS.Optimizer, verbose = false)
# set_lp_solver!(pm, solver)

# training objectivec
lambda = 1f-2 

# logitbinarycrossentropy(logŷ, y) = (1 .- y).*logŷ .- logσ.(logŷ)
# objective(x, y) = mean(logitbinarycrossentropy(model(x), y)) + lambda * sum(norm, params(model))
objective(x, y) = ap_objective(model(x), y, pm) + lambda * sum(norm, params(model))

accuracy(x, y) = mean((model(x) .>= 0.0f0) .== y)
evaluate(x, y) = compute_metric(pm, Int.((model(x) .>= 0.0f0) |> cpu), y |> cpu)
evaluate_cs(x, y) = compute_constraints(pm, Int.((model(x) .>= 0.0f0) |> cpu), y |> cpu)


# optimizer
eta = 0.001
opt = ADAM(eta)

val = evaluate(validation_set...)
val_tr = evaluate(X_tr |> gpu, y_tr |> gpu)
@info(@sprintf("[0]: Validation metric: %.4f, Train metric: %.4f", val, val_tr))

if pm.info.n_constraints > 0
    cs = evaluate_cs(validation_set...)
    cs_tr = evaluate_cs(X_tr |> gpu, y_tr |> gpu)
    @info(@sprintf("[0]: Validation cs: %.4f, Train cs: %.4f", cs[1], cs_tr[1]))
end


@info("Beginning training loop...")
best_val = 0.0
last_improvement = 0
model_filename = dname * "-" * string(pm) * "-l" * string(lambda) * "-e" * string(eta) * ".bson"
for epoch_idx in 1:100
    global best_val, last_improvement

    # Train for a single epoch
    Flux.train!(objective, params(model), train_set, opt)

    # Calculate metric:
    val = evaluate(validation_set...)
    val_tr = evaluate(X_tr |> gpu, y_tr |> gpu)
    @info(@sprintf("[%d]: Validation metric: %.4f, Train metric: %.4f", epoch_idx, val, val_tr))
    
    # calculate constraints
    if pm.info.n_constraints > 0
        cs = evaluate_cs(validation_set...)
        cs_tr = evaluate_cs(X_tr |> gpu, y_tr |> gpu)

        @info(@sprintf("[%d]: Validation cs: %.4f, Train cs: %.4f", epoch_idx, cs[1], cs_tr[1]))
    end
   
    # If this is the best metric we've seen so far, save the model out
    if pm.info.n_constraints == 0 
        if val >= best_val
            @info(" -> New best metric! Saving model out to " * model_filename)
            model_cpu = model |> cpu
            BSON.@save joinpath(dirname(@__FILE__), "model", model_filename) model_cpu epoch_idx
            best_val = val
            last_improvement = epoch_idx
        end
    else
        desired_cs = [pm.data.constraint_list[i].threshold for i = 1:pm.info.n_constraints]
        # if this is the best metric and the constraints is within 10% of desired cs
        if val >= best_val && all(cs .>= 0.95 * desired_cs)
            @info(" -> New best metric! Saving model out to " * model_filename)
            model_cpu = model |> cpu
            BSON.@save joinpath(dirname(@__FILE__), "model", model_filename) model_cpu epoch_idx
            best_val = val
            last_improvement = epoch_idx
        end
    end

end


@info("End of training. Evaluating on test set")
@info("Load the best saved model")

# load the best model saved
BSON.@load joinpath(dirname(@__FILE__), "model", model_filename) model_cpu
model = model_cpu |> gpu

# Calculate metric:
val_ts = evaluate(test_set...)
val_tr = evaluate(X_tr |> gpu, y_tr |> gpu)
@info(@sprintf("[%s]: Test metric: %.4f, Train metric: %.4f", dname, val_ts, val_tr))

# calculate constraints
if pm.info.n_constraints > 0
    cs_ts = evaluate_cs(test_set...)
    cs_tr = evaluate_cs(X_tr |> gpu, y_tr |> gpu)

    @info(@sprintf("[%s]: Test cs: %.4f, Train cs: %.4f", dname, cs_ts[1], cs_tr[1]))
end

@info("DONE")