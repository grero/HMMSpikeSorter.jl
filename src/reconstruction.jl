function reconstruct_signal(x::Array{T,1}, lA::StateMatrix, μ::Array{Float64,2}, σ::Float64) where T <: Integer
    Y2 = zeros(Float64, length(x))
    for i in 1:length(x)
        for j in 1:lA.N
            Y2[i] += μ[lA.states[j,x[i]], j]
        end
    end
    Y2
end

