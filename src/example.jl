using PyPlot
include("HMMSpikeSorter.jl")

temp1 = HMMSpikeSorter.create_spike_template(60,3.0, 0.8, 0.2)
temp2 = HMMSpikeSorter.create_spike_template(60,4.0, 0.3, 0.2)
temps = cat(2, temp1, temp2)

S = HMMSpikeSorter.create_signal(20_000, 0.3, [0.003, 0.001], temps)
f = plt[:figure]()
ax1 = f[:add_subplot](121)
ax2 = f[:add_subplot](122)
ax1[:plot](S;label="Original")
ax2[:plot](temps;label="Original spikeforms")

function update_plot(x)
    ax2[:clear]()
    ax2[:plot](temps;label="Original spikeforms")
    ax2[:plot](x;label="Fitted templates")
    ax2[:legend]()
end

lA, μ, σ = HMMSpikeSorter.train_model(S, 3, 60, false, 10,update_plot)
μ,uidx = HMMSpikeSorter.remove_small(μ, σ)
qidx = setdiff(1:lA.N, uidx)
lp,sidx = HMMSpikeSorter.get_lp(lA)
pidx = find(x->all(lA.states[qidx,x].==1), 1:lA.nstates)
lA2 = HMMSpikeSorter.StateMatrix(length(uidx), 60, lp[uidx], lA.π[pidx],lA.resolve_overlaps)

lA2, μ_new, σ = HMMSpikeSorter.train_model(S, lA2, μ_new, σ)
update_plot(μ_new)

x,T2, T1 = HMMSpikeSorter.viterbi(S, lA2, μ_new, σ)
Y2 = HMMSpikeSorter.reconstruct_signal(x, lA2, μ_new, σ)

ax1[:plot](Y2;label="Reconstructed")

ax1[:legend]()

