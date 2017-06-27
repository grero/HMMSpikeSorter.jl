function StatsBase.fit(::Type{HMMSpikingModel}, X::Array{Float64,1}, N=3, K=60,nsteps=10,resolve_overlaps=false, callback::Function=x->nothing)
    templates = fit(HMMSpikeTemplateModel, X, N, K, nsteps, resolve_overlaps, callback)
    fit(HMMSpikingModel, templates, X, callback)
end

function StatsBase.fit(::Type{HMMSpikingModel}, templates::HMMSpikeTemplateModel, X::Array{Float64,1},  callback::Function=x->nothing;kvs...)
    x,T2, T1 = viterbi(X, templates.state_matrix, templates.μ, templates.σ)
    #compute log-likelihood
    ll = 0.0
    for i in 1:length(x)
        ll += T1[x[i],i]
    end
    HMMSpikingModel(templates, x,ll,X)
end

function StatsBase.fit(::Type{HMMSpikingModel}, templates::HMMSpikeTemplateModel, X::Array{Float64,1},  chunksize::Integer, callback::Function=x->nothing)
    i = 1
    j = 1
    n = length(X)
    ml_seq = ones(Int16,n)
    ll = 0.0
    while j < n
        j = min(i + chunksize-1,n)
        k = j-i+1
        x,T2, T1 = viterbi(view(X,i:j), templates.state_matrix, templates.μ, templates.σ)
        while x[k] > 1
            j -= 1
            k -= 1
        end
        ml_seq[i:j] = x[1:k]
        for i in 1:k
            ll += T1[x[i],i]
        end
        i = j
    end
    HMMSpikingModel(templates, ml_seq, ll, X)
end

function StatsBase.fit(::Type{HMMSpikeTemplateModel}, X::Array{Float64,1}, N=3, K=60,nsteps=10,resolve_overlaps=false, callback::Function=x->nothing;kvs...)
    lA, μ, σ = train_model(X, N, K, resolve_overlaps, nsteps, callback;kvs...)
    HMMSpikeTemplateModel(lA, μ, σ)
end

function StatsBase.fit(model::HMMSpikeTemplateModel, X::AbstractArray{Float64,1}, nsteps, callback)
    lA, μ, σ = train_model(X, model.state_matrix, model.μ, model.σ, nsteps,callback)
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

