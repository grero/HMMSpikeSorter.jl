
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

create_spike_template(nstates::Integer) = create_template(nstates, 1.0, 0.8, 0.2)
function create_spike_template(nstates::Integer, a::Real, b::Real, c::Real)
    x = linspace(0,1.5,nstates)
    y = a*sin(2*pi*x).*exp(-(b-x).^2/c)
    return y
end

function create_signal(N::Integer, sigma::Number, pp::Array{Float64,1}, templates::Array{Float64,2})
    nstates,ncells = size(templates)
    state = zeros(Int64, ncells)
    S = rand(Normal(0,sigma),N)
    active_cell = 0
    for i in 1:length(S)
        if active_cell == 0
            for j in 1:ncells
                r = rand()
                if pp[j] > r
                    state[j] = 1
                    active_cell = j
                    break
                else
                    active_cell = 0
                    state[j] = 0
                end
            end
        end
        if active_cell > 0
            S[i] += templates[state[active_cell], active_cell]
            state[active_cell] += 1
            if state[active_cell] > nstates
                state[active_cell] = 0
                active_cell = 0
            end
        end
    end
    return S
end

