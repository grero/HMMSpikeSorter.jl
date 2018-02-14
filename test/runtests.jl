using Base.Test
using HMMSpikeSorter
using StatsBase
srand(0)

#temp1 = HMMSpikeSorter.create_spike_template(60,3.0, 0.8, 0.2)
#temp2 = HMMSpikeSorter.create_spike_template(60,4.0, 0.3, 0.2)
#temps = cat(2, temp1, temp2)
#S = HMMSpikeSorter.create_signal(20_000, 0.3, [0.003, 0.001], temps)
#lA, μ, σ = HMMSpikeSorter.train_model(S, 3, 60, false, 5;verbose=0)
#x,ll = HMMSpikeSorter.viterbi(S, lA, μ, σ)
#Y2 = HMMSpikeSorter.reconstruct_signal(x, lA, μ, σ)
#
#@test_approx_eq 1-std(Y2-S)/std(S) 0.5764879059017742

function test_viterbi()
    temp1 = HMMSpikeSorter.create_spike_template(60,3.0, 0.8, 0.2)
    temp2 = HMMSpikeSorter.create_spike_template(60,4.0, 0.3, 0.2)
    temps = cat(2, temp1, temp2)
    pp = [0.003, 0.001]
    S = HMMSpikeSorter.create_signal(20_000, 0.3, pp, temps)
    lA = HMMSpikeSorter.StateMatrix(2, 60, log.(pp), true)
    templates = HMMSpikeSorter.HMMSpikeTemplateModel(lA, temps, 0.3)
    modelf = HMMSpikeSorter.fit(HMMSpikeSorter.HMMSpikingModel, templates, S)
    Y = predict(modelf)
    1-std(Y-S)./std(S)
end

aa = test_viterbi()
@test aa ≈ 0.5753193438558597

@testset "Unroll" begin
    state_matrix = HMMSpikeSorter.StateMatrix(2,5,log.([0.01, 0.004]))
    mlseq = Int16[1 1 1 2 3 4 5 1 6 7 8 9 1 10 15 20 25 1][:]
    _mlseq = HMMSpikeSorter.unroll_mlseq(mlseq, state_matrix)
    @test _mlseq[1,:] == [1,1,1,2,3,4,5,1,1,1,1,1,1,2,3,4,5,1]
    @test _mlseq[2,:] == [1,1,1,1,1,1,1,1,2,3,4,5,1,2,3,4,5,1]
end
