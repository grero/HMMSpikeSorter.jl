module HMMSpikeSorter
using StatsBase
import StatsBase.fit, StatsBase.predict
using Distributions
include("utils.jl")
include("types.jl")
include("baumwelch.jl")
include("viterbi.jl")
include("reconstruction.jl")
include("fit.jl")
end #module
