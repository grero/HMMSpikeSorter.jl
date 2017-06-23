module HMMSpikeSorter
using StatsBase
using FileIO
using MAT
import StatsBase.fit, StatsBase.predict, StatsBase.loglikelihood, StatsBase.model_response
import FileIO.save
using Distributions
include("utils.jl")
include("types.jl")
include("baumwelch.jl")
include("viterbi.jl")
include("reconstruction.jl")
include("fit.jl")
include("extraction.jl")
end #module
