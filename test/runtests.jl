using Test
using HMMSpikeSorter
using StatsBase
using Random
using Statistics

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
    rng = MersenneTwister(UInt32(1234))
    temp1 = HMMSpikeSorter.create_spike_template(60,3.0, 0.8, 0.2)
    temp2 = HMMSpikeSorter.create_spike_template(60,4.0, 0.3, 0.2)
    temps = cat(temp1, temp2, dims=2)
    pp = [0.003, 0.001]
    S = HMMSpikeSorter.create_signal(20_000, 0.3, pp, temps;rng=rng)
    lA = HMMSpikeSorter.StateMatrix(2, 60, log.(pp), true)
    templates = HMMSpikeSorter.HMMSpikeTemplateModel(lA, temps, 0.3)
    modelf = HMMSpikeSorter.fit(HMMSpikeSorter.HMMSpikingModel, templates, S)
    Y = predict(modelf)
    1-std(Y-S)./std(S)
end

@testset "Viterbi" begin
    aa = test_viterbi()
    @test 0.55 < aa < 0.57
end

@testset "Unroll" begin
    state_matrix = HMMSpikeSorter.StateMatrix(2,5,log.([0.01, 0.004]))
    mlseq = Int16[1 1 1 2 3 4 5 1 6 7 8 9 1 10 15 20 25 1][:]
    _mlseq = HMMSpikeSorter.unroll_mlseq(mlseq, state_matrix)
    @test _mlseq[1,:] == [1,1,1,2,3,4,5,1,1,1,1,1,1,2,3,4,5,1]
    @test _mlseq[2,:] == [1,1,1,1,1,1,1,1,2,3,4,5,1,2,3,4,5,1]
end

@testset "overlap and combine" begin
    μ = permutedims([[1.0] [2.0] [3.0];[1.0] [2.0] [3.0]],[2,1])
    xi,xm = HMMSpikeSorter.find_best_overlap(μ, 1,2)
    @test xi == (1:3, 1:3)
    @test xm ≈ 14.0
    temp1 = HMMSpikeSorter.create_spike_template(60,3.0, 0.8, 0.2)
    t2 = fill!(similar(temp1), 0.0)
    t2[:5:end] = temp1[1:56]
    xi,xm = HMMSpikeSorter.find_best_overlap(cat(temp1,t2, dims=2),1,2)
    @test xi[1] == 1:56
    @test xi[2] == 5:60
    @test xm ≈ 100.66411692920131

    candidates, test_stat, overlap_idx = HMMSpikeSorter.condense_templates(cat(temp1, t2,dims=2), 0.1)
    @test candidates == (1,2)
    @test overlap_idx[1] == 1:56
    @test overlap_idx[2] == 5:60
end

@testset "match templates" begin
    μ = permutedims([[1.0] [2.0] [3.0];[1.0] [2.0] [3.0]],[2,1])
    μ[:,1] .*= 1.3
    xi, xm = HMMSpikeSorter.match_templates(μ, μ)
    @test xi == [1,2]
    @test xm ≈ [0.0, 0.0]
end

@testset "Baum-Welch" begin
    rng = MersenneTwister(UInt32(1234))
    temp1 = HMMSpikeSorter.create_spike_template(60,3.0, 0.8, 0.2)
    temp2 = HMMSpikeSorter.create_spike_template(60,4.0, 0.3, 0.2)
    temps = cat(temp1, temp2, dims=2)
    pp = [0.003, 0.001]
    S = HMMSpikeSorter.create_signal(30_000, 0.3, pp, temps;rng=rng)
    templates = StatsBase.fit(HMMSpikeSorter.HMMSpikeTemplateModel, S, 7)
    @test size(templates[2],2) == 2
    midx, ms = HMMSpikeSorter.match_templates(temps, templates[2])
    @test ms[1]/sum(abs2, temps[:,1]) < 0.01
    @test ms[2]/sum(abs2, temps[:,2]) < 0.01
end

@testset "Noise energy" begin
    rng = MersenneTwister(UInt32(1234))
    temp1 = HMMSpikeSorter.create_spike_template(60,3.0, 0.8, 0.2)
    temp2 = HMMSpikeSorter.create_spike_template(60,4.0, 0.3, 0.2)
    temps = cat(temp1, temp2, dims=2)
    pp = [0.003, 0.001]
    S = HMMSpikeSorter.create_signal(30_000, 0.3, pp, temps;rng=rng)
    EE = HMMSpikeSorter.get_noise_energy(S, 1.0/(0.3*0.3), 60;rng=rng)
    @test 66.0 < EE < 66.7
end
