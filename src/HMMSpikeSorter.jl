__precompile__()

module HMMSpikeSorter
using StatsBase
using FileIO
using MAT
using ProgressMeter
import FileIO.save
import StatsBase.fit, StatsBase.predict, StatsBase.loglikelihood, StatsBase.model_response, StatsBase.bic
using Distributions
include("utils.jl")
include("types.jl")
include("baumwelch.jl")
include("viterbi.jl")
include("reconstruction.jl")
include("fit.jl")
include("extraction.jl")
end #module
