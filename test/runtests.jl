using Base.Test
using HMM
srand(0)

temp1 = HMM.create_spike_template(60,3.0, 0.8, 0.2)
temp2 = HMM.create_spike_template(60,4.0, 0.3, 0.2)
temps = cat(2, temp1, temp2)
S = HMM.create_signal(20_000, 0.3, [0.003, 0.001], temps)
lA, μ, σ = HMM.train_model(S, 3, 60, false, 5,update_plot;verbose=0)
x,T2, T1 = HMM.viterbi(S, lA, μ, σ)
Y2 = HMM.reconstruct_signal(x, lA, μ, σ)

@test_approx_eq 1-std(Y2-S)/std(S) 0.5764879059017742

