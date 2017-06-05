module HMM
using Distributions
include("utils.jl")
include("types.jl")
include("baumwelch.jl")
include("viterbi.jl")
include("reconstruction.jl")
end #module
