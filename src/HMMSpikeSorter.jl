__precompile__()

module HMMSpikeSorter
using StatsBase
using FileIO
using MAT
using Printf
using ProgressMeter
import FileIO.save
import StatsBase.fit, StatsBase.predict, StatsBase.loglikelihood, StatsBase.model_response, StatsBase.bic
import Base.isempty
using Distributions
using Random
include("utils.jl")
include("types.jl")
include("baumwelch.jl")
include("viterbi.jl")
include("fit.jl")
include("extraction.jl")
end #module
