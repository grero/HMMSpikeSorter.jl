
func(x, μ, σ) = (σ2 = σ*σ; dd=x-μ;exp(-dd*dd/(2*σ2))/sqrt(2*pi*σ2))
funcl(x, μ, σ) = (σ2 = σ*σ; dd=x-μ;-0.5*log(2*pi*σ2) - dd*dd/(2*σ2))

"""
Logpdf of log-normal distribution
"""
function func2l(x,μ, σ) 
    σ2 = σ*σ
    dd = log(x) - μ
    return -0.5*log(2*pi*σ2) - log(x) - dd*dd/(2*σ2)
end

"""
Computes log(x+y) in a numerically stable way
"""
function logsumexp(x,y)
   xp = log(x)
   yp = log(y)
   logsumexpl(xp, yp)
end

function logsumexpl(xp,yp)
   z = 0.0
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

function generate_states(N,K)
    states = zeros(Int64,N,1+N*(K-1) + N*(N-1)*(K-1)*(K-1))
    k = 2
    for i in 1:N
        for k1 in 1:K-1
            states[i,k] = k1
            k += 1
        end
    end
    for i in 1:N
       for j in 1:N
           if i==j
               continue
           end
           for k1 in 1:K-1
               for k2 in 1:K-1
                   states[i,k] = k1
                   states[j,k] = k2
                   k += 1
               end
           end
       end
   end
   states
end

function isvalid_transition(states,K,lp,j1,j2)
    lpt = 0.0
    for i in 1:size(states,1)
        s1 = states[i,j1]
        s2 = states[i,j2]
        lpi = lp[i]
        if s1 == s2 == 0
            lpt += log1p(-exp(lpi))
        elseif s1 == 0 && s2 == 1
            lpt += lpi
        elseif (s2 - s1 == 1) || (s1 == K-1 && s1 == K-1)
            lpt += 0.0
        else #other transitions simply add 0.0
            lpt = -Inf #impossible transition
            break
        end
    end
    lpt
end

function get_valid_transitions(states::Array{Int64,2}, K,lp)
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
