function forward(V::Array{Float64,1},lpp::Array{Float64,1}, lA::Array{Float64,2}, μ::Array{Float64,1}, σ::Float64)
    T = length(V)
    nstates = size(lA,1)
    a = zeros(nstates, T)
    for i=1:nstates
        a[i,1] = lpp[i] + func2l(V[1],μ[i], σ)
    end
    aa = 0.0
    for i=2:T
        for j=1:nstates
            b = func2l(V[i], μ[j], σ)
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
    for i in 1:nstates
        a[i,1] = lA.π[i]
        ss = lA.states[:,i]
        for j in 1:lA.N
            a[i,1] += func2l(V[1],μ[ss[j],j], σ)
        end
    end
    states = lA.states
    N = lA.N
    for i=2:T
        v = V[i]
        for qq in lA.transitions
            k = qq[1]
            j = qq[2]
            lp = qq[3]
            b = 0.0
            for l in 1:N
                b += func2l(v, μ[states[l,j],l], σ)
            end
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
                    b = func2l(V[i+1], μ[k], σ)
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
    a = log(zeros(nstates, T))
    a[:,T] = 0.0
    for i in T-1:-1:1
        v = V[i+1]
        for qq in lA.transitions
            j = qq[1]
            k = qq[2]
            lp = qq[3]
            b = 0.0
            for l in 1:N
                b += func2l(v, μ[states[l,k], l], σ)
            end
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
            g = logsumexpl(g, α[j,t]+β[j,t] )
		end
		for j in 1:nstates
			γf[j,t] = α[j,t] + β[j,t] - g
		end
	end
    #println(all(isfinite(γf)))
	for t in 1:length(x) - 1
		q = -Inf
		for j in 1:nstates
			for i in 1:nstates
                if isfinite(lA[j,i]) #ignore impossible transitions
                    q = logsumexpl(q, α[j,t]+lA[j,i]+β[i,t+1]+func2l(x[t+1],μ[i],σ))
                end
			end
		end
		for j in 1:nstates
			for i in 1:nstates
                if isfinite(lA[j,i])
                    ξ[i,j,t] = α[j,t]+lA[j,i]+β[i,t+1]+func2l(x[t+1],μ[i],σ) - q
                end
			end
		end
	end
	#TODO: Check this
	#Anew = squeeze(sum(ξ[:,:,1:end-1],3)./sum(γf[:,1:end-1],3),3)
    Anew = log(eye(nstates))
    Anew = log(diagm(ones(nstates-1),1))
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
    @show Anew[1,1], bb
    Anew[1,1] -= bb[1]
    Anew[1,2] = log1p(-exp(Anew[1,1])) #compute log(1-exp(a))
    Anew[end,1] = 0.0
    Anew[end,end] = -Inf

	#TODO: Check if this actually works
	_σ = zeros(μ)
    gg = log(zeros(μ))
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
	for t in 1:length(x)
		g = -Inf
		for j in 1:nstates
            g = logsumexpl(g, α[j,t]+β[j,t] )
		end
		for j in 1:nstates
			γf[j,t] = α[j,t] + β[j,t] - g
		end
	end
    #update transition matrix; the only non-determnisitc transitions are from noise (state 0) to active for each neuron
    sidx = find(sum(lA.states.==2,1).==1) #find these transition states for each neuron
    ξ = zeros(lA.N+1, length(x)-1)
    for t in 1:length(x)-1
        _x = x[t+1]
        lp = lA.transitions[1][3]
        #note μ[1,:] should be zero
        ξ[1,t] = α[1,t] + lp + β[1,t+1] + func2l(_x, μ[1,1], σ)
        for i in 1:N
            j = sidx[i]
            #find the transition from states 1 to state j
            tidx = findfirst(q->q[1]==1 && q[2]==j, lA.transitions)
            lp = lA.transitions[tidx][3]
            ξ[i+1,t] = α[1,t]  + lp + β[j,t+1] + func2l(_x, μ[lA.states[i,j],i],σ)
        end
        q = -Inf
        for i in 1:N+1
            q = logsumexpl(q, ξ[i,t])
        end
        for i in 1:N+1
            ξ[i,t] -= q
        end
    end
    #println(extrema(sum(exp(ξ),1)))
    #println(extrema(sum(exp(γf),1)))
    #println(γf[:,1])
    #println(ξ[:,1])
    bb = -Inf
    xx = log(zeros(N+1))
    for t in 1:length(x)-1
        bb = logsumexpl(bb, γf[1,t]) #we need only the silent state
        for j in 1:N+1
            xx[j] = logsumexpl(xx[j],ξ[j,t])
        end
    end
    #update the transition matrix with the new transitions
    pp = γf[:,1]
    @show xx, bb
    xb = xx-bb #FIXME: This does not work!
    #the problem here is that xx is usually higher than bb, which means that we see more transitions out of a given state than we see occupations of that state. This is clearly non-sensical, and indicates a bug somewhere.
    #xb = min(0.0, xb)
    lA_new = StateMatrix(lA.states-1, pp, K, xb[2:end];allow_overlaps=lA.resolve_overlaps)
	_σ = zeros(μ)
    gg = log(zeros(μ))
    x2 = 0.0
    qq = 0.0
	for j in 1:nstates
		x1 = -Inf
        #only look at states where a single neuron is active
        aidx =  find(lA.states[:,j].>=2)
        #FIXME: Check that the μ and σ are correctly computed
        if length(aidx) == 1
            _aidx = aidx[1]
            ss = lA.states[_aidx,j]
            for t in 1:length(x)
                #convert to log-normal
                _x = log(x[t])
                #eγf = exp(γf[j,t])
                eγf = γf[j,t]
                x1 = logsumexpl(x1, eγf+_x)
                gg[ss,_aidx] = logsumexpl(gg[ss,_aidx], eγf)
            end
            if ss > 1 #only upgade the mean for other states
                μ[ss,_aidx] = exp(x1-gg[ss,_aidx]) # the log of the mean of the log-normal distributed variable
            end
            #μ[ss,_aidx] = x1/exp(gg[ss,_aidx])
            #_σ[ss, _aidx] = exp(x2-gg[ss,_aidx]) - μ[ss, _aidx]*μ[ss,_aidx]
            #@show _σ[ss, _aidx]
        end
        #upgrade variance; assumed equal for for all neurons and states
        for t in 1:length(x)
            _x = log(x[t])
            eγf = γf[j,t]
            for i in 1:N
                d = (x[t]-μ[lA.states[i,j],i])
                x2 += d*d*exp(eγf) 
                qq += exp(eγf)
            end
        end
	end
    σ2 = x2/qq 
    println(extrema(γf))
    #σ2 = sum(exp(gg).*_σ)/sum(exp(gg))
    #σ2 = exp(x2)
    println(σ2)
    σ = sqrt(σ2)
	#END TODO
    lA_new, μ, σ
end

function train_model(X,N=3,K=60, resolve_overlaps=false, nsteps=100,callback::Function=x->nothing)
    lp = log(fill(0.1, N))
    state_matrix = StateMatrix(N,K,lp, resolve_overlaps) 
    μ = exp(randn(K,N))
    μ[1,:] = 1.0 #all neurons must start from silence
    σ = log(0.1)
	for i in 1:nsteps
        print("$i ")
		state_matrix, μ, σ = train_model(X, state_matrix, μ, σ)
        callback(μ)
        yield()
	end
    println()
	state_matrix, μ, σ
end

function train_model(X,pp0, aa0, μ0, σ0)
	α = forward(X, pp0, aa0, μ0, σ0)
	β = backward(X, aa0, μ0, σ0)
	pp,aa,μ,σ = update(α, β, aa0, μ0, σ0, X)
end

function train_model(X,state_matrix, μ0, σ0)
	α = forward(X, state_matrix, μ0, σ0)
	β = backward(X, state_matrix, μ0, σ0)
	state_matrix,μ,σ = update(α, β, state_matrix, μ0, σ0, X)
end

function decode()
end


function prepA(p,n)
    A = log(zeros(n,n))
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
