function viterbi(y::Array{Float64,1}, lA::Array{Float64,2}, lpp::Array{Float64,1}, μ::Array{Float64,1}, σ::Float64)
    #straight forward implementation of the Viterbi algorithm
    #assume gaussian emission probabilities
    nstates = size(lA,1)
    nstates == size(μ,1) || throw(ArgumentError("size(B,2) should be equal to the number of states"))
    nobs = size(y,1)
    x = zeros(Int16, nobs)
    T1 = log(zeros(Float64, nstates, nobs))
    T2 = zeros(Int16, nstates, nobs)
    #define initial elements
    for i=1:nstates
        T1[i,1] = lpp[i]+func2l(y[1], μ[i], σ)
    end
    #T1[1,1] = 0
    for i=2:nobs
        for j=1:nstates    
            tm = -Inf
            km = 0
            q = func2l(y[i], μ[j], σ)
            for k=1:nstates
                #probability of transitioning from state k to state j and emitting observation i
                t = T1[k,i-1]+lA[k,j]+q
                #j==1 && k==60 && println(string("1->60 $t"))
                #j==1 && k==1 && println(string("1->1 $t"))
                if t > tm
                    tm = t
                    km = k
                end
            end
            T1[j,i] = tm
            T2[j,i] = km
        end
    end
    #define the last state
    #mx,x[end] = findmax(T1[:,end])
    x[end] = 1
    #run backward
    for i=nobs:-1:2
        x[i-1] = T2[x[i],i]
    end
    return x,T2, T1
end

function viterbi(y::AbstractArray{Float64,1}, lA::StateMatrix, μ::Array{Float64,2}, σ::Float64)
    #straight forward implementation of the Viterbi algorithm
    #assume gaussian emission probabilities
    lσ = log(σ)
    nstates = lA.nstates
    N = lA.N
    nobs = size(y,1)
    x = zeros(Int16, nobs)
    T1 = fill(-Inf, nstates, nobs)
    T2 = ones(Int16, nstates, nobs)
    #define initial elements
    @inbounds for i=1:nstates
        T1[i,1] = lA.π[i]
        _μ = 0.0
        for j in 1:N
            _μ += μ[lA.states[j,i],j]
        end
        T1[i,1] = funcl(y[1], _μ, σ, lσ)
    end
    T1[1,1] = 0
    q = zeros(nstates)
    @inbounds for i=2:nobs
        yi = y[i]
        for j in 1:nstates
            _μ = 0.0
            for l in 1:N
                _μ += μ[lA.states[l,j],l]
            end
            q[j] = funcl(yi, _μ, σ, lσ)
        end
        for qq in lA.transitions
            k = qq[1]
            j = qq[2]
            lp = qq[3]

            t = T1[k,i-1]+lp
            if t > T1[j,i] 
                T1[j,i] = t
                T2[j,i] = k
            end
        end
        for j in 1:nstates
            T1[j,i] += q[j]
        end
    end
    #define the last state
    x[end] = indmax(T1[:,end])
    #run backward
    ll = 0.0
    for i=nobs:-1:2
        @inbounds x[i-1] = T2[x[i],i]
        ll += T1[x[i], i]
    end
    return x,ll
end

function createIndex(N,K)
    #create a linear index for N rings, each with K states
    npairs = div(N*(N-1),2)
    Q = zeros(Int64,npairs,K,K)
    KK = (K-1)*(K-1)
    NK = N*(K-1)+1
    NP = NK + KK
    i = 1
    for k1=1:N-1
        for k2=k1+1:N
            Q[i,2:end,2:end] = reshape([1:KK]+NK + (i-1)*NP, (1,K-1,K-1))
            Q[i,1,:] = [1:K] + (k1-1)*(K-1)
            Q[i,2:end,1] = [2:K] + (k2-1)*(K-1)
            i += 1
        end
    end
    return Q
end

function unfoldIndex(states,stateMatrix)
    npairs,nstates,nstates = size(stateMatrix)
    ncells = div(int(1+sqrt(1+4*2*npairs)),2) #get the number of cells
    #cheat
    pairs = zeros(Int16,ncells,ncells)
    k = 1
    for k1=1:ncells-1
        for k2=k1+1:ncells
            pairs[k1,k2] = k
            k+=1
        end
    end
    #pairs = pairs + pairs'
    X = zeros(Int64,length(states),ncells)
    for i=1:length(states)
        pair,state1,state2 = ind2sub((npairs,nstates,nstates), find(stateMatrix.==states[i]))
        cell1,cell2 = ind2sub((ncells,ncells),find(pairs.==pair[1]))
        X[i,cell1] = state1[1]
        X[i,cell2] = state2[1]
    end
    return X-1
end
