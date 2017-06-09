function StatsBase.fit(::Type{HMMSpikingModel}, X::Array{Float64,1}, N=3, K=60,nsteps=10,resolve_overlaps=false, callback::Function=x->nothing)
    templates = fit(HMMSpikeTemplateModel, X, N, K, nsteps, resolve_overlaps, callback)
    fit(HMMSpikingModel, templates, X, callback)
end

function StatsBase.fit(::Type{HMMSpikingModel}, templates::HMMSpikeTemplateModel, X::Array{Float64,1},  callback::Function=x->nothing)
    x,T2, T1 = viterbi(X, templates.state_matrix, templates.μ, templates.σ)
    HMMSpikingModel(templates, x)
end

function StatsBase.fit(::Type{HMMSpikeTemplateModel}, X::Array{Float64,1}, N=3, K=60,nsteps=10,resolve_overlaps=false, callback::Function=x->nothing)
    lA, μ, σ = train_model(X, N, K, resolve_overlaps, nsteps, callback)
    HMMSpikeTemplateModel(lA, μ, σ)
end

function StatsBase.predict(model::HMMSpikingModel)
    reconstruct_signal(model.ml_seq, model.template_model.state_matrix, model.template_model.μ, model.template_model.σ)
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

