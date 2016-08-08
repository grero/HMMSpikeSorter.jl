using Distributions

function forward(V::Array{Float64,1},A::Array{Float64,2}, mu::Array{Float64,1}, sigma::Real)
    T = length(V)
    nstates = size(A,1)
    a = zeros(nstates, T)
    a[1,1] = 1
    LP = Array(Normal, nstates)
    for i=1:nstates
        LP[i] = Normal(mu[i],sigma)
    end
    aa = 0.0
    Z = 0.0
    for i=2:T
        Z = 0.0
        for j=1:nstates
            b = pdf(LP[j],V[i])
            aa = 0.0
            for k=1:nstates
                aa += a[k,i-1]*A[k,j]
            end
            a[j,i] = b*aa
            Z += a[j,i]
        end
        for j=1:nstates
            a[j,i] /= Z
        end
    end
    return a
end

function backward(V::Array{Float64,1},A::Array{Float64,2}, mu::Array{Float64,1}, sigma::Real)
    T = length(V)
    nstates = size(A,1)
    a = zeros(nstates, T)
    a[1,T] = 1
    LP = Array(Normal, nstates)
    for i=1:nstates
        LP[i] = Normal(mu[i],sigma)
    end
    aa = 0.0
    Z = 0.0
    for i=T-1:-1:1
        Z = 0.0
        for j=1:nstates
            aa = 0.0
            for k=1:nstates
                b = pdf(LP[k],V[i+1])
                aa += a[k,i+1]*A[j,k]
            end
            a[j,i] = aa 
            Z += a[j,i]
        end
        for j=1:nstates
            a[j,i] /= Z
        end
    end
    return a
end

function gamma(α,β,A,μ,σ,x)
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

function update(α, β, A, μ, σ, x)
	nstates = size(A,1)
	LP = Array(Function, nstates)
	for i=1:nstates
		LP[i] = x->pdf(Normal(μ[i],σ),x)
	end
	γ = zeros(length(x), nstates)
	ξ = zeros(length(x), nstates, nstates)
	for t in 1:length(x)
		g = 0.0
		for j in 1:nstates
			g += α[j,t]*β[j,t] 
		end
		for j in 1:nstates
			γ[t,j] = α[j,t]*β[j,t]/g
		end
	end
	for t in 1:length(x) - 1
		q = 0.0
		for j in 1:nstates
			for i in 1:nstates
				q += α[j,t]*A[j,i]*β[i,t+1]*LP[i](x[t+1])
			end
		end
		for j in 1:nstates
			for i in 1:nstates
				ξ[t,j,i] = α[j,t]*A[j,i]*β[i,t+1]*LP[i](x[t+1])
			end
		end
	end
	Anew = squeeze(sum(ξ[1:end-1,:,:],1)./sum(γ[1:end-1,:],1),1)
	#renormalize
	Anew = Anew./sum(Anew,2)
	#TODO: Check if this actually works
	_σ = zeros(μ)
	gg = zeros(μ)
	for j in 1:nstates
		x1 = 0.0
		x2 = 0.0
		for t in 1:length(x)
			x1 += γ[t,j]*x[t]
			x2 += γ[t,j]*x[t]*x[t]
			gg[j] += γ[t,j]
		end
		μ[j] = x1./gg[j]
		_σ[j] += x2./gg[j] - μ[j].*μ[j]
	end
	σ = sqrt(sum(gg.*_σ)/sum(gg))
	#END TODO
	γ[1,:], Anew, μ, σ
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
