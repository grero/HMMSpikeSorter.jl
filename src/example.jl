using PyPlot
include("HMM.jl")

temp1 = HMM.create_spike_template(60,3.0, 0.8, 0.2)
temp2 = HMM.create_spike_template(60,4.0, 0.3, 0.2)
temps = cat(2, temp1, temp2)

S = HMM.create_signal(20_000, 0.3, [0.003, 0.001], temps)
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

lA, μ, σ = HMM.train_model(S, 3, 60, false, 10,update_plot)
x,T2, T1 = HMM.viterbi(S, lA, μ, σ)
Y2 = HMM.reconstruct_signal(x, lA, μ, σ)

ax1[:plot](Y2;label="Reconstructed")

ax1[:legend]()

