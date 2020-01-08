module HMMSpikeSorter
using StatsBase
using FileIO
using MAT
using ProgressMeter
using Printf
import FileIO.save
import StatsBase.fit, StatsBase.predict, StatsBase.loglikelihood, StatsBase.model_response, StatsBase.bic
import Base.isempty
using Distributions
include("utils.jl")
include("types.jl")
include("baumwelch.jl")
include("viterbi.jl")
include("reconstruction.jl")
include("fit.jl")
include("extraction.jl")
end #module
