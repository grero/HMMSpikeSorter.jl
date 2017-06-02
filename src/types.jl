type StateMatrix
    states::Array{Int64,2}
    transitions::Array{Tuple{Int64,Int64,Float64},1}
    π::Array{Float64,1}
    K::Int64 #number of neurons
    N::Int64 #number of states per neuron
    nstates::Int64
end

function generate_states(N,K,allow_overlaps=true)
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


function StateMatrix(states::Array{Int64,2},pp::Array{Float64,1}, K,lp)
    transitions = get_valid_transitions(states,K,lp)
    StateMatrix(states+1, transitions,pp, K,size(states,1),size(states,2))
end
