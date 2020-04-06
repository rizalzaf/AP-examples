# AP-Examples

This repository provides code examples for the experiments in an AISTATS 2020 paper ["AP-Perf: Incorporating Generic Performance Metrics in Differentiable Learning"](https://arxiv.org/abs/1912.00965) by [Rizal Fathony](http://rizal.fathony.com) and [Zico Kolter](http://zicokolter.com).


## Required packages

To run the code, some pre-requisite packages need to be installed. Below is the list of the packages:
```
AdversarialPrediction
Flux@0.10
StatsBase
DelimitedFiles
BSON
LoggingExtras
EvalCurves
``` 

The `AdversarialPrediction` package can be installed from [https://github.com/rizalzaf/AdversarialPrediction.jl](https://github.com/rizalzaf/AdversarialPrediction.jl) by running `] add AdversarialPrediction`. For the `Flux` package, we require `Flux v0.10` or later, which uses Zygote-based auto differentiation tool. The `EvalCurves` can be installed from a Julia terminal using the command: `]add https://github.com/vitskvara/EvalCurves.jl`. All other packages can be installed from a Julia terminal using the command: `]add PackageName`.

To run the experiments on GPU, the following packages also need to be installed.
```
CuArrays
CUDAdrv
CUDAnative
``` 

## Running the codes

The codes for the tabular data experiments are available in `tabular.jl` and `tabular_bce.jl` for the experiment with adversarial prediction and cross-entropy objectives respectively. To run the adversarial prediction experiment, just run `julia tabular.jl dataset metric` command from the console where `dataset` is the dataset name and `metric` can be any of the following: `acc`, `f1`, `f2`, `gpr`, `mcc`, `kappa`, `pr8`, and `pr6`. The list of available datasets is in the `data-cv` folder. Below are some examples of the command for running the experiments.

```
julia tabular.jl whitewine f1
julia tabular.jl whitewine f2
julia tabular.jl censusdomains gpr

julia tabular_bce.jl whitewine
julia tabular_bce.jl censusdomains
```

The experiments with image datasets follows similar patterns.
```
julia mnist.jl f1
julia fmnist.jl f2
julia fmnist.jl gpr

julia mnist_bce.jl
julia fmnist_bce.jl
```