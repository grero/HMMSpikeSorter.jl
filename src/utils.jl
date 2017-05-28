
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
