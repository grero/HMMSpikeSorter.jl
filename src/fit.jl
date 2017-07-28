function StatsBase.fit(::Type{HMMSpikingModel}, X::Array{Float64,1}, N=3, K=60,nsteps=10,resolve_overlaps=false, callback::Function=x->nothing)
    templates = fit(HMMSpikeTemplateModel, X, N, K, nsteps, resolve_overlaps, callback)
    fit(HMMSpikingModel, templates, X, callback)
end

function StatsBase.fit(::Type{HMMSpikingModel}, templates::HMMSpikeTemplateModel, X::Array{Float64,1},  callback::Function=x->nothing;kvs...)
    x,ll = viterbi(X, templates.state_matrix, templates.μ, templates.σ)
    HMMSpikingModel(templates, x,ll,X)
end

function StatsBase.fit(::Type{HMMSpikingModel}, templates::HMMSpikeTemplateModel, X::Array{Float64,1},  chunksize::Integer, callback::Function=x->nothing)
    i = 1
    j = 1
    n = length(X)
    ml_seq = ones(Int16,n)
    ll = 0.0
    pp = Progress(n)
    while j < n
        gc()
        j = min(i + chunksize-1,n)
        k = j-i+1
        l = 1
        x,_ll  = viterbi(view(X,i:j), templates.state_matrix, templates.μ, templates.σ)
        if i > 1 #make sure we start from silence
            #if not, just skip those sections
            while x[l] > 1
                l += 1
            end
        end
        if j < n
            while x[k] > 1
                j -= 1
                k -= 1
            end
        end
        ml_seq[(i+l-1):j] .= x[l:k]
        ll += _ll
        i = j
        update!(pp, i)
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

function StatsBase.bic(model::HMMSpikingModel)
    k = length(model.template_model.μ) + 1 + model.template_model.state_matrix.N
    n = length(model.ml_seq)
    log(n)*k - 2*loglikelihood(model)
end

