using EvalCurves

function prec_at_rec(x, y, rec = 0.8)
    ps = model(x) |> cpu
    prec_at_rec(data(ps), y |> cpu, rec)
end

function prec_at_rec(ps::AbstractVector{<:Number}, y::AbstractVector{<:Number}, rec = 0.8)

    recvec, precvec = EvalCurves.prcurve(ps, y)
    id = findfirst(x -> x >= rec, recvec)

    return precvec[id]
end