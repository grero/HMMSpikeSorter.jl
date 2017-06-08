function StatsBase.fit(::Type{HMMSpikingModel}, X::Array{Float64,1}, N=3, K=60,nsteps=10,resolve_overlaps=false, callback::Function=x->nothing)
    lA, μ, σ = train_model(X, N, K, resolve_overlaps, nsteps, callback)
    x,T2, T1 = viterbi(X, lA, μ, σ)
    HMMSpikingModel(lA, x, μ, σ)
end

function StatsBase.predict(model::HMMSpikingModel)
    reconstruct_signal(model.ml_seq, model.state_matrix, model.μ, model.σ)
end

function plot(model::HMMSpikingModel)
    f = plt[:figure]()
    ax1 = f[:add_subplot](121)
    ax2 = f[:add_subplot](122)
    Y2 = reconstruct_signal(model.ml_seq, model.state_matrix, model.μ, model.σ)
    ax1[:plot](Y2)
    ax2[:plot](μ)
    f
end

