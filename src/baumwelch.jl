function forward(V::Array{Float64,1},lpp::Array{Float64,1}, lA::Array{Float64,2}, μ::Array{Float64,1}, σ::Float64)
    T = length(V)
    nstates = size(lA,1)
    a = zeros(nstates, T)
    for i=1:nstates
        a[i,1] = lpp[i] + funcl(V[1],μ[i], σ)
    end
    aa = 0.0
    for i=2:T
        for j=1:nstates
            b = funcl(V[i], μ[j], σ)
            aa = -Inf
            for k=1:nstates
                #aa += a[k,i-1]*A[k,j]
                if isfinite(lA[k,j]) #ignore impossible transtions
                    aa = logsumexpl(aa, a[k,i-1] + lA[k,j])
                end
            end
            a[j,i] = b+aa
        end
    end
    return a
end

function forward(V::Array{Float64,1}, lA::StateMatrix, μ::Array{Float64,2}, σ::Float64)
    T = length(V)
    nstates = lA.nstates
    a = fill(-Inf,nstates, T)
    _μ = zeros(nstates)
    for i in 1:nstates
        a[i,1] = lA.π[i]
        ss = lA.states[:,i]
        for j in 1:lA.N
            _μ[i] += μ[ss[j],j]
        end
        a[i,1] = funcl(V[1],_μ[i], σ)
    end
    states = lA.states
    N = lA.N
    for i=2:T
        v = V[i]
        for qq in lA.transitions
            k = qq[1]
            j = qq[2]
            lp = qq[3]
            b = funcl(v, _μ[j], σ)
            a[j,i] = logsumexpl(a[j,i], a[k,i-1] + lp + b)
        end
    end
    a
end

function backward(V::Array{Float64,1},lA::Array{Float64,2}, μ::Array{Float64,1}, σ::Float64)
    T = length(V)
    nstates = size(lA,1)
    a = zeros(nstates, T)
    aa = 0.0
    for i=T-1:-1:1
        for j=1:nstates
            aa = -Inf
            for k=1:nstates
                if isfinite(lA[j,k])
                    b = funcl(V[i+1], μ[k], σ)
                    aa = logsumexpl(aa, a[k,i+1] + lA[j,k]+ b)
                end
            end
            a[j,i] = aa
        end
    end
    return a
end

function backward(V::Array{Float64,1},lA::StateMatrix, μ::Array{Float64,2}, σ::Float64)
    T = length(V)
    nstates = lA.nstates
    K = lA.K
    N = lA.N
    states = lA.states
    a = fill(-Inf, nstates, T)
    a[:,T] = 0.0
    _μ = zeros(nstates)
    for j in 1:nstates
        for l in 1:N
            _μ[j] += μ[states[l,j], l]
        end
    end
    for i in T-1:-1:1
        v = V[i+1]
        for qq in lA.transitions
            j = qq[1]
            k = qq[2]
            lp = qq[3]
            b = funcl(v, _μ[k], σ)
            a[j,i] = logsumexpl(a[j,i], a[k, i+1] + lp + b)
        end
    end
    a
end

function fgamma(α,β,A,μ,σ,x)
    nstates = size(A,1)
		n = length(x)
    G = zeros(nstates,nstates,n)
    LP = Array(Normal, nstates)
    for i=1:nstates
        LP[i] = Normal(μ[i],σ)
    end
    for t=2:n
        for j=1:nstates
            for i=1:nstates
                G[i,j,t] = α[i,t-1]*A[i,j]*β[j,t]*pdf(LP[j],x[t])
            end
        end
    end
    return G
end


function update(α, β, lA, μ, σ, x)
	nstates = size(lA,1)
	γf = zeros(nstates, length(x))
	ξ = zeros(nstates, nstates, length(x))
	for t in 1:length(x)
		g = -Inf
		for j in 1:nstates
            @inbounds g = logsumexpl(g, α[j,t]+β[j,t] )
		end
		for j in 1:nstates
			@inbounds γf[j,t] = α[j,t] + β[j,t] - g
		end
	end
    #println(all(isfinite(γf)))
	for t in 1:length(x) - 1
		q = -Inf
		for j in 1:nstates
			for i in 1:nstates
                @inbounds if isfinite(lA[j,i]) #ignore impossible transitions
                    @inbounds q = logsumexpl(q, α[j,t]+lA[j,i]+β[i,t+1]+funcl(x[t+1],μ[i],σ))
                end
			end
		end
		for j in 1:nstates
			for i in 1:nstates
                @inbounds if isfinite(lA[j,i])
                    @inbounds ξ[i,j,t] = α[j,t]+lA[j,i]+β[i,t+1]+funcl(x[t+1],μ[i],σ) - q
                end
			end
		end
	end
	#TODO: Check this
	#Anew = squeeze(sum(ξ[:,:,1:end-1],3)./sum(γf[:,1:end-1],3),3)
    Anew = log.(eye(nstates))
    Anew = log.(diagm(ones(nstates-1),1))
    Anew[1,1] = -Inf
    for t in 1:length(x)-1
        #for i in 1:nstates
        #    for j in 1:nstates
        #        Anew[j,i] = logsumexpl(Anew[j,i], ξ[j,i,t])
        #    end
        #end
        Anew[1,1] = logsumexpl(Anew[1,1], ξ[1,1,t])
    end

    bb = zeros(nstates)
    for t in 1:length(x)-1
        for i in 1:nstates
            bb[i] = logsumexpl(bb[i], γf[i,t])
        end
    end
    #for i in 1:nstates
        #for j in 1:nstates
        #    Anew[j,i] -= bb[j]
        #end
    #end
    Anew[1,1] -= bb[1]
    Anew[1,2] = log1p(-exp(Anew[1,1])) #compute log(1-exp(a))
    Anew[end,1] = 0.0
    Anew[end,end] = -Inf

	#TODO: Check if this actually works
	_σ = zeros(μ)
    gg = fill(-Inf,size(mu))
	for j in 1:nstates
		x1 = -Inf
		x2 = -Inf
		for t in 1:length(x)
            #convert to log-normal
            _x = log(x[t])
            #eγf = exp(γf[j,t])
            eγf = γf[j,t]
            x1 = logsumexpl(x1, eγf+_x)
            x2 = logsumexpl(x2, eγf+_x+_x)
            gg[j] = logsumexpl(gg[j], eγf)
		end
        μ[j] = exp(x1-gg[j]) # the log of the mean of the log-normal distributed variable
        _σ[j] = exp(x2-gg[j]) - μ[j]*μ[j]
	end
    σ = sqrt(sum(exp(gg).*_σ)/sum(exp(gg)))
	#END TODO
    pp = γf[:,1]
    pp, Anew, μ, σ
end

##KIND OF  WORKS##
function update(α::Array{Float64,2}, β::Array{Float64,2}, lA::StateMatrix, μ::Array{Float64,2}, σ::Float64, x::Array{Float64,1})
    nstates = lA.nstates
    N = lA.N
    K = lA.K
    γf = zeros(nstates, length(x))
    _μ = zeros(nstates)
    for j in 1:nstates
        for l in 1:N
            @inbounds _μ[j] += μ[lA.states[l,j],l]
        end
    end
	for t in 1:length(x)
		g = -Inf
		for j in 1:nstates
            @inbounds g = logsumexpl(g, α[j,t]+β[j,t] )
		end
		for j in 1:nstates
			@inbounds γf[j,t] = α[j,t] + β[j,t] - g
		end
	end
    #update transition matrix; the only non-determnisitc transitions are from noise (state 0) to active for each neuron
    tidx = find(q->q[1]==1, lA.transitions) #all transitions out of state 1
    #Note: we can also have all neurons transitioning from silence to active
    ξ = zeros(length(tidx), length(x)-1)
    for t in 1:length(x)-1
        _x = x[t+1]
        #note μ[1,:] should be zero
        #ξ[1,t] = α[1,t] + lp + β[1,t+1] + funcl(_x, μ[1,1], σ)
        for i in 1:length(tidx)
            #j = sidx[i]
            #find the transition from states 1 to state j
            #tidx = findfirst(q->q[1]==1 && q[2]==j, lA.transitions)
            @inbounds j = lA.transitions[tidx[i]][2]
            @inbounds lp = lA.transitions[tidx[i]][3]
            @inbounds bb = funcl(_x, _μ[j],σ)
            @inbounds ξ[i,t] = α[1,t]  + lp + β[j,t+1] + bb
        end
        q = -Inf
        for qq in lA.transitions
            i = qq[1]
            j = qq[2]
            lp = qq[3]
            @inbounds bb = funcl(_x, _μ[j] ,σ)
            @inbounds q = logsumexpl(q,α[i,t]  + lp + β[j,t+1] + bb)
        end
        for i in 1:size(ξ,1)
            @inbounds ξ[i,t] -= q
        end
    end
    bb = -Inf
    xx = fill(-Inf, size(ξ,1))
    for t in 1:length(x)-1
        @inbounds bb = logsumexpl(bb, γf[1,t]) #we need only the silent state
        for j in 1:size(ξ,1)
            @inbounds xx[j] = logsumexpl(xx[j],ξ[j,t])
        end
    end
    #update the transition matrix with the new transitions
    pp = γf[:,1]
    xb = xx-bb
    lA_new = StateMatrix(lA.states-one(Int16), pp, K, xb[2:end];allow_overlaps=lA.resolve_overlaps)
	_σ = zeros(μ)
    gg = zeros(μ)
    fill!(μ, 0.0)
    sidx = find(sum(lA.states.>=2,1).==1) #find states with only one active neuron
    for t in 1:length(x)
        _x = x[t]
        for j in sidx
            @inbounds eγf = exp(γf[j,t])
            for l in 1:N
                @inbounds ss = lA.states[l,j]
                if ss > 1
                    @inbounds μ[ss, l] += _x*eγf
                    @inbounds gg[ss,l] += eγf
                end
            end
        end
    end
    for l in 1:N
        for j in 2:K
            @inbounds μ[j,l] /= gg[j,l]
        end
    end
    fill!(_μ, 0.0)
    for j in 1:nstates
        for l in 1:N
            @inbounds _μ[j] += μ[lA.states[l,j],l]
        end
    end
    #upgrade variance; assumed equal for for all neurons and states
    x2 = 0.0
    qq = 0.0
    for t in 1:length(x)
        @inbounds for j in 1:nstates
            _x = x[t]
            eγf = exp(γf[j,t])
            d = _x-_μ[j]
            x2 += d*d*eγf 
            qq += eγf
        end
	end
    σ2 = x2/qq 
    σ = sqrt(σ2)
    lA_new, μ, σ
end

function train_model(X,N::Integer=3,K::Integer=60, resolve_overlaps=false, nsteps::Integer=8,callback::Function=x->nothing;verbose::Integer=0,p0=2.0^(-3*K/2))
    lp = log.(fill(p0, N))
    state_matrix = StateMatrix(N,K,lp, resolve_overlaps) 
    μ = ones(K,N)
    σ = std(X)
    for i in 1:N
        μ[:,i] = create_spike_template(K, 3*σ*rand(), 0.5+0.1*randn(), 1.5*rand())
    end
    #μ = exp(rand(K,N))
    μ[1,:] = 0.0 #all neurons must start from silence
    train_model(X, state_matrix, μ, σ, nsteps, callback;verbose=verbose)
end

function train_model(X,state_matrix::StateMatrix, μ::Array{Float64,2}, σ::Float64, nsteps::Integer,callback::Function=x->nothing;verbose::Integer=1)
	for i in 1:nsteps
        if verbose > 0
            print("$i ")
        end
        callback(μ)
        yield()
		state_matrix, μ, σ = train_model(X, state_matrix, μ, σ;verbose=verbose)
        if isempty(state_matrix)
            break
        end
	end
    if verbose > 0
        println()
    end
    #TODO: Merge templates that are similar first
    state_matrix, μ = condense_templates(state_matrix, μ, σ, 0.05;verbose=verbose)
    verbose > 0 && println("$(size(μ,2)) templates remain after merging")
    #prune
    verbose > 0 && println("Removing sparse templates...")
    state_matrix, idx = remove_sparse(state_matrix)
    verbose > 0 && println("$(length(idx)) templates remain after removing sparse")
    verbose > 0 && println("Removing small templates...")
    state_matrix, idx2 = remove_small(state_matrix, μ[:,idx], σ, StatsBase.PValue(0.05))
    verbose > 0 && println("$(length(idx2)) templates remain after removing small templates")
    μ = μ[:,idx[idx2]]
    for i in 1:div(nsteps,2)
		state_matrix, μ, σ = train_model(X, state_matrix, μ, σ;verbose=verbose)
    end
    state_matrix, μ, σ
end

function train_model_old(X,pp0, aa0, μ0, σ0)
	α = forward(X, pp0, aa0, μ0, σ0)
	β = backward(X, aa0, μ0, σ0)
	pp,aa,μ,σ = update(α, β, aa0, μ0, σ0, X)
end

function train_model(X::Array{Float64,1},state_matrix::StateMatrix, μ0::Array{Float64,2}, σ0::Float64;verbose=0)
    verbose > 0 && println("Running forward algorithm...")
	α = forward(X, state_matrix, μ0, σ0)
    verbose > 0 && println("Running backward algorithm...")
	β = backward(X, state_matrix, μ0, σ0)
    verbose > 0 && println("Running update algorithm...")
	state_matrix,μ,σ = update(α, β, state_matrix, μ0, σ0, X)
    state_matrix,  μ, σ
end

function decode()
end


function prepA(p,n)
    A = log.(zeros(n,n))
    A[1,1] = log(1.-p)
    A[1,2] = log(p)
    for i=2:n-1
        A[i,i+1] = 0
    end
    A[n,1] = 0
    return A
end

function test()
    A = prepA(1e-9)
end

function simulate(pp, aa, μ, σ,n)
	X = zeros(n)	
	states = Array(Int64,n)
	PP = cumsum(pp[:])
	AA = cumsum(aa,2)
	state = searchsortedfirst(PP, rand())
	X[1] = rand(Normal(μ[state], σ))
	states[1] = state
	for i in 2:n
		state = searchsortedfirst(AA[state,:][:],rand()) #transition
		X[i] = rand(Normal(μ[state], σ))
		states[i] = state
	end
	X,states
end

function test(S,lp, lA,μ, σ)
    lAl = prepA(exp(lp[1]),60)
    α = forward(exp(S), lA.π, lAl, μ[:,1], σ);
    β = backward(exp(S), lAl, μ[:,1], σ);
    a = forward(exp(S), lA, μ[:,1:1], σ);
    b = backward(exp(S), lA, μ[:,1:1], σ);
    lAp, μp, σp = update(a, b, lA, μ[:,1:1], σ, exp(S))
    ppn, lAn, μn, σn = update(α, β, lAl, μ[:,1], σ, exp(S))
    lAp, lAn
end

"""
Removes templates from `μ` that are significantly not different from noise at a p-value of `α`.

    function remove_small(μ::Array{Float64,2}, σ::Float64,α=0.05)
"""
function remove_small(state_matrix::StateMatrix, μ::Array{Float64,2}, σ::Float64,  α::StatsBase.PValue=StateBase.Pvalue(0.05))
    nstates = size(μ,1)
    σ2 = σ.*σ
    Z = sum(μ.^2,1)./σ2
    #use the fact that Z is Χ² distributed with n-1 degress of freedom
    pvals = 1-cdf.(Chisq(nstates-1),Z)
    tidx = find(pvals .< α.v)
    lA = prune_templates(state_matrix, tidx, state_matrix.resolve_overlaps)
    lA, tidx
end

function remove_small(templates::HMMSpikeTemplateModel, α=0.05,resolve_overlaps=true)
    idx = remove_small(templates.μ, templates.σ, templates.state_matrix)
    prune_templates(templates, idx, resolve_overlaps)
end

function condense_templates(templates::HMMSpikeTemplateModel, α=0.05)
    μ = templates.μ
    K,N = size(μ)
    σ² = templates.σ^2
    condense_templates(templates.state_matrix, μ, σ, α)
end

function condense_templates(state_matrix::StateMatrix, μ::Matrix{Float64}, σ::Float64, α::Float64=0.05;verbose=0)
    σ² = σ^2
    lp,_idx = get_lp(state_matrix)
    K,N = size(μ)
    candidates, test_stat, overlap_idx = condense_templates(μ, σ², α)
    while !isempty(candidates)
        i1,i2 = candidates
        xi1, xi2 = overlap_idx
        if verbose > 1
            println("Merging templates $i1 and $i2 with Χ² statistic $test_stat")
        end
        N -= 1
        μ_new = zeros(K,N)
        lp_new = zeros(N)
        kk = 1
        μ_new[xi1,kk] = 0.5.*μ[xi1,i1]
        μ_new[xi2, kk] .=+ 0.5*μ[xi2,i2]
        lp_new[1] = log(0.5*exp(lp[i1]) + 0.5*exp(lp[i2]))
        idx = setdiff(1:N,[i1,i2])
        for (ii,jj) in enumerate(idx)
            μ_new[:,kk+ii] = μ[:,jj]
            lp_new[kk+ii] = lp[jj]
        end
        lp = lp_new
        μ = μ_new
        candidates, test_stat, overlap_idx = condense_templates(μ, σ², α)
    end
    if N < state_matrix.N
        state_matrix_new = StateMatrix(N, K, lp, state_matrix.resolve_overlaps) 
        return state_matrix_new, μ
    end
    return state_matrix, μ
end

function condense_templates(μ::Matrix{Float64}, σ²::Real, α=0.05)
    K,N = size(μ)
    candidates = Vector{Tuple{Int64,Int64}}(0)
    test_stat = Float64[]
    overlap_idx = Vector{NTuple{2, UnitRange{Int64}}}(0)
    for i1 in 1:N-1
        for i2 in i1+1:N
            xi,xm = find_best_overlap(μ, i1,i2)
            x = 0.0
            for (k1,k2) in zip(xi...)
                x += abs2(μ[k1,i1]-μ[k2,i2])
            end
            x /= σ²
            #heuristic; if fewer than 5 points match, do not consider this a match
            if length(xi[1]) < 5
                pval = 0.0
            else
                pval = 1-cdf(Chisq(length(xi[1])-1),x)
            end
            if pval > α #merge if the distance is compatible with noise
                push!(candidates, (i1,i2))
                push!(test_stat, x)
                push!(overlap_idx, xi)
            end
            #compute θ = sum((t1-t2)Y2/σ
            #merge if the difference is comptabible with noise
        end
    end
    #merge the most similar templates first
    if !isempty(candidates)
        merge_idx = indmax(test_stat)
        return candidates[merge_idx], test_stat[merge_idx], overlap_idx[merge_idx] 
    end
    return candidates, test_stat, overlap_idx
end

"""
Finds the best overlap between templates indexed by `i1` and `i2`.
"""
function find_best_overlap(μ::Matrix{Float64}, i1::Int64, i2::Int64)
    K,N = size(μ)
    xi = (1:K, 1:K)
    xm = -Inf 
    shifts = [(1:s, K-s+1:K) for s in 1:K]
    append!(shifts, [(1+s:K, 1:K-s) for s in 1:K-1])
    for shift in shifts
        x = 0.0
        y1 = 0.0
        y2 = 0.0
        for (k1,k2) in zip(shift...)
            x += μ[k1,i1]*μ[k2,i2]
            y1 += μ[k1,i1]^2
            y2 += μ[k2,i2]^2
        end
        if x > xm
            xm = x
            xi = shift
        end
    end
    xi,xm
end

"""
Match the templates from `temps1` to those in `temps2` by comparing sum of pointwise square differences at the optimum alignment 
"""
function match_templates(temps1::Matrix{Float64}, temps2::Matrix{Float64})
    K1,N1 = size(temps1)
    K2,N2 = size(temps2)
    K1 == K2 || throw(ArgumentError("The two template sets must have the same number of states"))
    K = K1
    mm = fill(0,N1)
    cc = fill(0.0, N1)
    for i1 in 1:N1
        m = Inf
        mi = 0
        for i2 in 1:N2
            #FIXME: Don't concatenate here
            xi,xm = find_best_overlap(cat(2,temps1[:,i1], temps2[:,i2]), 1,2)
            xm  = sum(abs2, temps1[xi[1],i1]-temps2[xi[2],i2])
            if xm < m
                m = xm
                mi = i2
            end
        end
        mm[i1] = mi
        cc[i1] = m
    end
    mm, cc
end

"""
Remove templates associated with very low firing rate.
"""
function remove_sparse(state_matrix::StateMatrix, lp0=-70.0;verbose=1)
    tt = filter(x->(x[1]==1) && (x[2] != 1) && (x[3] > lp0), state_matrix.transitions)
    if isempty(tt)
        return StateMatrix(), Int64[]
    end
    #get the state we transition to
    idx = [_tt[2] for _tt in tt]
    #get the active templates for these states
    tidx = Int64[]
    for (i,ix) in enumerate(idx)
        for j in 1:size(state_matrix.states,1)
            if state_matrix.states[j,ix] == 2
                push!(tidx, j)
                break
            end
        end
    end
    lA = prune_templates(state_matrix, tidx, state_matrix.resolve_overlaps)
    lA, tidx
end

function remove_small(state_matrix::StateMatrix, mu::Matrix{Float64}, sigma::Float64, data::AbstractVector{T};theshold=4.0) where T<: Real
    een = get_noise_energy(data, 1.0/sigma, nstates) 
    remove_small(state_matrix, mu, sigma, een;threshold=threshold)
end

function remove_small(state_matrix::StateMatrix, mu::Matrix{Float64}, sigma::Float64, ee::Float64;threshold=4)
    nstates = size(mu,1)
    eef = get_noise(mu, 1.0/sigma)
    tidx = find(q->q>threshold*ee,eef)
    lA = prune_templates(state_matrix, tidx, state_matrix.resolve_overlaps)
    lA, tidx
end
