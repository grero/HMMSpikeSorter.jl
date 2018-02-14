"""
Unroll the 1-D `mlseq`, returning a 2-D array with the state of each neuron
"""
function unroll_mlseq(mlseq::AbstractVector{Int16}, state_matrix::StateMatrix)
    _mlseq = zeros(Int16, state_matrix.N, length(mlseq))
    for i in 1:length(mlseq)
        mi = mlseq[i]
        for j in 1:state_matrix.N
            _mlseq[j,i] = state_matrix.states[j,mi]
        end
    end
    _mlseq
end

function extract_spiketimes(model::HMMSpikingModel)
    pidx = Array{Array{Int64,1}}(0)
    for i in 1:model.template_model.state_matrix.N
        qidx = indmin(model.template_model.μ[:,i])
        sidx = find(model.template_model.state_matrix.states[i,:] .== qidx)
        _pidx = findin(model.ml_seq,sidx)
        push!(pidx, _pidx)
    end
    pidx
end

function extract_units(model::HMMSpikingModel, channel::Integer;sampling_rate=40000.0)
    pidx = extract_spiketimes(model::HMMSpikingModel)
    units = Dict()
    for i in 1:length(pidx)
        unit_name = @sprintf "g%03dc%02d_spiketrain.mat" channel i
        units[unit_name] = Dict("timestamps" => pidx[i],
                                "sampling_rate" => sampling_rate,
                                "waveform" => model.template_model.μ[:,i])
    end
    units
end

function save_units(units::Dict)
    for (k,v) in units
        MAT.matwrite("sorted/$k", v)
    end
end

function load_unit(fname)
    MAT.matread(fname)
end
