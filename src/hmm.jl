using Distributions

func(x, μ, σ) = (σ2 = σ*σ; dd=x-μ;exp(-dd*dd/(2*σ2))/sqrt(2*pi*σ2))
funcl(x, μ, σ) = (σ2 = σ*σ; dd=x-μ;-0.5*log(2*pi*σ2) - dd*dd/(2*σ2))

"""
Computes log(x+y) in a numerically stable way
"""
function logsumexp(x,y)
   z = 0.0
   xp = log(x)
   yp = log(y)
   logsumexpl(xp, yp)
end

function logsumexpl(xp,yp)
   if xp > yp
      z = xp + log1p(exp(yp-xp))
   else
      z = yp + log1p(exp(xp-yp))
   end
   return z
end

function logsumexpl{T<:Real}(X::Array{T,1},i=1)
    n = length(X)
    if i == n-1 
        return logsumexpl(X[i], X[i+1]) 
    else
        a = X[i]
        yp = logsumexpl(X, i+1)
        if a > yp
            return a + log1p(exp(yp - a))
        else
            return yp + log1p(exp(a - yp))
        end
     end
end

function forward(V::Array{Float64,1},pp::Array{Float64,1}, A::Array{Float64,2}, μ::Array{Float64,1}, σ::Float64)
    T = length(V)
    nstates = size(A,1)
    a = zeros(nstates, T)
    for i=1:nstates
        a[i,1] = pp[i]*func(V[1],μ[i], σ)
    end
    aa = 0.0
    Z = 0.0
    for i=2:T
        Z = 0.0
        for j=1:nstates
            b = func(V[i], μ[j], σ)
            aa = 0.0
            for k=1:nstates
                aa += a[k,i-1]*A[k,j]
            end
            a[j,i] = b*aa
            Z += a[j,i]
        end
    end
    return a
end

function backward(V::Array{Float64,1},A::Array{Float64,2}, μ::Array{Float64,1}, σ::Float64)
    T = length(V)
    nstates = size(A,1)
    a = ones(nstates, T)
    aa = 0.0
    Z = 0.0
    for i=T-1:-1:1
        Z = 0.0
        for j=1:nstates
            aa = 0.0
            for k=1:nstates
                b = func(V[i+1], μ[k], σ)
                aa += a[k,i+1]*A[j,k]*b
            end
            a[j,i] = aa + eps()
            Z += a[j,i]
        end
        #for j=1:nstates
        #    a[j,i] /= Z
        #end
    end
    return a
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


function update(α, β, A, μ, σ, x)
	nstates = size(A,1)
	γf = zeros(nstates, length(x))
	ξ = zeros(nstates, nstates, length(x))
	for t in 1:length(x)
		g = 0.0
		for j in 1:nstates
			g += α[j,t]*β[j,t] 
		end
		for j in 1:nstates
			γf[j,t] = α[j,t]*β[j,t]/g
		end
	end
	for t in 1:length(x) - 1
		q = 0.0
		for j in 1:nstates
			for i in 1:nstates
                q += α[j,t]*A[j,i]*β[i,t+1]*func(x[t+1],μ[i],σ)
			end
		end
		for j in 1:nstates
			for i in 1:nstates
                ξ[i,j,t] = α[j,t]*A[j,i]*β[i,t+1]*func(x[t+1],μ[i],σ)
			end
		end
	end
	#TODO: Check this
	#Anew = squeeze(sum(ξ[:,:,1:end-1],3)./sum(γf[:,1:end-1],3),3)
    Anew = squeeze(sum(ξ[:,:,1:end-1],3),3)./squeeze(sum(γf[:,1:end-1],2),2)
	#renormalize
	Anew = Anew./sum(Anew,2)
	#TODO: Check if this actually works
	_σ = zeros(μ)
	gg = zeros(μ)
	for j in 1:nstates
		x1 = 0.0
		x2 = 0.0
		for t in 1:length(x)
			x1 += γf[j,t]*x[t]
			x2 += γf[j,t]*x[t]*x[t]
			gg[j] += γf[j,t]
		end
		μ[j] = x1./gg[j]
		_σ[j] += x2./gg[j] - μ[j].*μ[j]
	end
	σ = sqrt(sum(gg.*_σ)/sum(gg))
	#END TODO
	γf[1,:][:], Anew, μ, σ
end

function train_model(X,nstates=2,nsteps=100)
	aa = ones(nstates,nstates)./nstates
	pp = ones(nstates)./nstates
	μ = zeros(nstates)
	σ = 1.0
	for i in 1:nsteps
		pp, aa, μ, σ = train_model(X, pp, aa, μ, σ)
	end
	pp, aa, μ, σ
end

function train_model(X,pp0, aa0, μ0, σ0)
	α = forward(X, pp0, aa0, μ0, σ0)
	β = backward(X, aa0, μ0, σ0)
	pp,aa,μ,σ = update(α, β, aa0, μ0, σ0, X)
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
