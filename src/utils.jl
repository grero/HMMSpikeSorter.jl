const log2pi = 0.5*log(2*pi)
func(x, μ, σ) = (σ2 = σ*σ; dd=x-μ;exp(-dd*dd/(2*σ2))/sqrt(2*pi*σ2))
funcl(x, μ, σ) = (σ2 = σ*σ; dd=x-μ;-log2pi-log(σ) - dd*dd/(2*σ2))
funcl(x, μ, σ, lσ) = (σ2 = σ*σ; dd=x-μ;-log2pi-lσ - dd*dd/(2*σ2))

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
    y = a*sin.(2*pi*x).*exp.(-(b-x).^2/c)
    return y
end

function create_signal(N::Integer, sigma::Number, pp::Array{Float64,1}, templates::Array{Float64,2};rng=MersenneTwister(rand(UInt32)))
    nstates,ncells = size(templates)
    state = zeros(Int64, ncells)
    S = sigma.*randn(rng, N) 
    active_cell = 0
    for i in 1:length(S)
        if active_cell == 0
            for j in 1:ncells
                r = rand(rng)
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

function get_chunk(X::AbstractArray{Float64,1},idx,chunksize=100_000)
    X[(idx-1)*chunksize+1:idx*chunksize]
end

"""
Estimate the noise energy of data by computing the normalized energy in `nsamples` random patches of length `nstates`.
"""
function get_noise_energy(data::AbstractVector{T1}, cinv::T2, nstates::Int64, nsamples=1000;rng=MersenneTwister(rand(UInt32))) where T1 <: Real where T2 <: Real
    N = length(data)
    samples = zeros(nsamples)
    idx = rand(rng, 1:N-nstates, nsamples)
    sort!(idx)
    for i in 1:nsamples
        @inbounds ii = idx[i]
        s = 0.0
        for j in 1:nstates
            x = data[ii+j-1]
            s += x*cinv*x
        end
        @inbounds samples[i] = s
    end
    return median(samples)
end

function get_energy(waveforms::Array{Float64,2}, cinv::Float64)
    nstates, N = size(waveforms)
    ee = zeros(N)
    for i in 1:N
        _ee = 0.0
        for j in 1:nstates
            @inbounds w = waveforms[j,i]
            _ee += w*cinv*w
        end
        @inbounds ee[i] = _ee
    end
    ee
end
