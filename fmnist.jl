using Flux, Flux.Data.FashionMNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf, BSON
using CuArrays
using CUDAdrv

using LinearAlgebra

using CUDAnative
CUDAnative.device!(0)

using Random
Random.seed!(0)

include("common_metrics.jl")

# CuArrays.allowscalar(false)

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

# Load labels and images from Flux.Data.FashionMNIST
@info("Loading data set")
train_labels = binarize(FashionMNIST.labels(), 0:0)
train_imgs = FashionMNIST.images()


# split training to train and validation
n_tr = Int.(round(0.7 * length(train_labels)))
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
validation_set = make_minibatch(val_imgs, val_labels, 1:length(val_imgs))
# gian train set for checking train metric
giant_train_set = make_minibatch(tr_imgs, tr_labels, 1:length(tr_imgs))

# Prepare test set as one giant minibatch:
test_imgs = FashionMNIST.images(:test)
test_labels = binarize(FashionMNIST.labels(:test), 0:0)
test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

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
giant_train_set = gpu.(giant_train_set)
validation_set = gpu.(validation_set)
test_set = gpu.(test_set)
model = gpu(model)

# metric
pm = f1_score
# pm = f2_score
# pm = gpr
# pm = inform
# pm = kappa
# pm = precision_gv_recall_80
# pm = precision_gv_recall_60
# pm = accuracy_metric

# training objectivec
lambda = 1f-2 
# logitbinarycrossentropy(logŷ, y) = (1 .- y).*logŷ .- logσ.(logŷ)
# objective(x, y) = mean(logitbinarycrossentropy(model(x), y)) + lambda * sum(norm, params(model))
objective(x, y) = ap_objective(model(x), y, pm) + lambda * sum(norm, params(model))

accuracy(x, y) = mean((model(x) .>= 0.0f0) .== y)
evaluate(x, y) = compute_metric(pm, Int.((model(x) .>= 0.0f0) |> cpu), y |> cpu)
evaluate_cs(x, y) = compute_constraints(pm, Int.((model(x) .>= 0.0f0) |> cpu), y |> cpu)


# Train our model with the given training set using the ADAM optimizer and
# printing out performance against the test set as we go.
eta = 0.001
opt = ADAM(eta)

val = evaluate(validation_set...)
val_tr = evaluate(giant_train_set...)
@info(@sprintf("[0]: Validation metric: %.4f, Train metric: %.4f", val, val_tr))

if pm.info.n_constraints > 0
    cs = evaluate_cs(validation_set...)
    cs_tr = evaluate_cs(giant_train_set...)
    @info(@sprintf("[0]: Validation cs: %.4f, Train cs: %.4f", cs[1], cs_tr[1]))
end

@info("Beginning training loop...")
best_val = 0.0
last_improvement = 0
model_filename = "FashionMNIST-" * string(pm) * "-l" * string(lambda) * "-e" * string(eta) * ".bson"
for epoch_idx in 1:100
    global best_val, last_improvement
    # Train for a single epoch
    Flux.train!(objective, params(model), train_set, opt)

    # Calculate metric:
    val = evaluate(validation_set...)
    val_tr = evaluate(giant_train_set...)
    @info(@sprintf("[%d]: Validation metric: %.4f, Train metric: %.4f", epoch_idx, val, val_tr))
    
    # calculate constraints
    if pm.info.n_constraints > 0
        cs = evaluate_cs(validation_set...)
        cs_tr = evaluate_cs(giant_train_set...)

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
val_tr = evaluate(giant_train_set...)
@info(@sprintf("[%s]: Test metric: %.4f, Train metric: %.4f", "FashionMNIST", val_ts, val_tr))

# calculate constraints
if pm.info.n_constraints > 0
    cs_ts = evaluate_cs(test_set...)
    cs_tr = evaluate_cs(giant_train_set...)

    @info(@sprintf("[%s]: Test cs: %.4f, Train cs: %.4f", "FashionMNIST", cs_ts[1], cs_tr[1]))
end

@info("DONE")