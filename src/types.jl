type StateMatrix
    states::Array{Int16,2}
    transitions::Array{Tuple{Int64,Int64,Float64},1}
    π::Array{Float64,1}
    K::Int64 #number of neurons
    N::Int64 #number of states per neuron
    nstates::Int64
    resolve_overlaps::Bool
end

type HMMSpikeTemplateModel
    state_matrix::StateMatrix
    μ::Array{Float64,2}
    σ::Float64
end

type HMMSpikingModel
    template_model::HMMSpikeTemplateModel
    ml_seq::Array{Int16,1}
    ll::Float64
    y::Array{Float64,1}
end

StatsBase.loglikelihood(model::HMMSpikingModel) = model.ll
StatsBase.model_response(model::HMMSpikingModel) = model.y

function HMMSpikingModel(lA::StateMatrix, ml_seq::Array{Int16,1}, μ::Array{Float64,2}, σ::Float64)
    HMMSpikingModel(HMMSpikeTemplateModel(lA, μ,σ), ml_seq)
end

"""
Get the transitions from noise to active for each neuron in `lA`.
"""
function get_lp(lA::StateMatrix)
    lp = zeros(lA.N)
    lidx = fill(0,lA.N)
    k = 1
    for (i,qq) in enumerate(lA.transitions)
        if qq[1] == 1 && qq[2]>1
            vidx =  find(lA.states[:,qq[2]].>1)
            if length(vidx) == 1 #only a single neuron active
                lp[k] = qq[3]
                lidx[k] = first(vidx)
                if k == lA.N
                    break
                else
                    k += 1
                end
            end
        end
    end
    lp,lidx
end

StateMatrix(states,transitions, Π, K, N, nstates) = StateMatrix(states, transitions, Π, K, N, nstates, true)

function generate_states(N,K,allow_overlaps=true)
    if allow_overlaps
        states = zeros(Int16,N,1+N*(K-1) + div(N*(N-1)*(K-1)*(K-1),2))
    else
        states = zeros(Int16,N,1+N*(K-1))
    end
    k = 2
    for i in 1:N
        for k1 in 1:K-1
            states[i,k] = k1
            k += 1
        end
    end
    if allow_overlaps
        for i in 1:N-1
           for j in i+1:N
               for k1 in 1:K-1
                   for k2 in 1:K-1
                       states[i,k] = k1
                       states[j,k] = k2
                       k += 1
                   end
               end
           end
       end
   end
   states
end

function isvalid_transition(states,K,lp,j1,j2)
    lpt = 0.0
    lpz = log1p(-exp(sum(lp))) #probability of staying in the silent state
    for i in 1:size(states,1)
        s1 = states[i,j1]
        s2 = states[i,j2]
        lpi = lp[i]
        if s1 == s2 == 0
            lpt += lpz
        elseif s1 == 0 && s2 == 1
            lpt += lpi
        elseif (s2 - s1 == 1) || (s1 == K-1 && s2 == 0)
            lpt += 0.0
        else #other transitions simply add 0.0
            lpt = -Inf #impossible transition
            break
        end
    end
    lpt
end

function get_valid_transitions(states::Array{Int16,2}, K,lp)
    idx = Array{Tuple{Int64, Int64, Float64}}(0)
    nstates = size(states,2)
    for i in 1:nstates
        for j in 1:nstates
            aa = isvalid_transition(states, K, lp,i,j)
            if isfinite(aa)
                push!(idx, (i,j,aa))
            end
        end
    end
    idx
end

function HMMSpikeTemplateModel(μ::Array{Float64,2}, lp::Array{Float64,1},σ = mean((μ[1,:]).^2), allow_overlaps=true)
    K,N = size(μ)
    state_matrix = StateMatrix(N,K, lp, allow_overlaps)
    HMMSpikeTemplateModel(state_matrix, μ, σ)
end

function StateMatrix(N::Int64,K::Int64, lp::Array{Float64,1},allow_overlaps::Bool=true)
    states = generate_states(N,K,allow_overlaps)
    nstates = size(states,2)
    pp = log.(ones(nstates)./nstates)
    StateMatrix(states, pp, K, lp;allow_overlaps=allow_overlaps)
end

function StateMatrix(N::Int64,K::Int64, lp::Array{Float64,1},pp::Array{Float64,1}, allow_overlaps::Bool=true)
    states = generate_states(N,K,allow_overlaps)
    nstates = size(states,2)
    StateMatrix(states, pp, K, lp;allow_overlaps=allow_overlaps)
end

function StateMatrix(states::Array{Int16,2},pp::Array{Float64,1}, K,lp;allow_overlaps=true)
    transitions = get_valid_transitions(states,K,lp)
    StateMatrix(states+1, transitions,pp, K,size(states,1),size(states,2),allow_overlaps)
end

"""
Create a new HMMSpikeTemplateModel with templates `idx`. Optionally, alllow these templates to overlap by setting `resolve_overlaps` to `true`.
"""
function prune_templates(templates::HMMSpikeTemplateModel, idx::AbstractArray{Int64,1},resolve_overlaps=true)
    lA = prune_templates(templates.state_matrix, idx, resolve_overlaps)
    HMMSpikeTemplateModel(lA, templates.μ[:,idx], templates.σ)
end

function prune_templates(state_matrix::StateMatrix, idx::AbstractVector{Int64}, resolve_overlaps=true)
    lp, tidx = get_lp(state_matrix)
    N = length(idx)
    lA = StateMatrix(N, state_matrix.K, lp[findin(tidx,idx)],resolve_overlaps)
    lA
end
