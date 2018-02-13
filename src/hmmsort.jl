using ArgParse
using HDF5
using MAT
using HMMSpikeSorter

function parseargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--inputfile"
        help = "File containing templates to be used for sorting"
        arg_type = String
        required = true
        default = ""
        "--datafile"
        help = "Data file containing data to bo sorted"
        arg_type = String
        required = true
        default = ""
        "--outputfile"
        help = "File to save the spiking model to"
        arg_type = String
        required = true
        default = ""
    end
    parsed_args = parse_args(ARGS, s)
end

function my_parse_args(args)
    parsed_args = Dict(zip(args[1:end-1], args[2:end]))
    for (k,v) in parsed_args
        parsed_args[strip(k, '-')] = v
    end
    return parsed_args
end

function sort_data(inputfile::String, datafile::String, outputfile::String;dosave=true, max_templates=4)
    #load template file 
    print("Loading templates...\n")
    template_model = HDF5.h5open(inputfile, "r") do ff
        waveforms = read(ff, "spikeForms")
        nstates,nchannels,ntemplates = size(waveforms)
        cinv = read(ff, "cinv")
        pp = read(ff, "p")
        if length(pp) > max_templates 
            state_matrix = HMMSpikeSorter.StateMatrix()
        else 
            state_matrix = HMMSpikeSorter.StateMatrix(ntemplates, nstates, log.(pp), true)
        end
        template_model = HMMSpikeSorter.HMMSpikeTemplateModel(state_matrix, waveforms[:,1,:], sqrt(inv(cinv[1])))
    end
    if length(template_model.state_matrix.states) == 1
        print("The number of templates exceeds the maximum. Bailing out...\n")
        return Dict()
    end
    print("Creating template model...\n")
    lp,ii = HMMSpikeSorter.get_lp(template_model.state_matrix)

    print("Loading data...\n")
    data = HDF5.h5open(datafile, "r") do ff
        if "rh" in names(ff)
            datapath = "rh/data/analogData"
        else
            datapath = "highpassdata/data/data"
        end
        if HDF5.ismmappable(ff[datapath])
            _data = HDF5.readmmap(ff[datapath])
        else
            _data = read(ff,datapath)
        end
        _data
    end
    if ndims(data) == 2
        _data = view(data, :, 1)
    else
        _data = data
    end
    if !(eltype(_data) == Float64)
        _dataf = convert(Vector{Float64}, _data)
    else
        _dataf = _Data
    end
    print("Fitting model...\n")
    modelf = HMMSpikeSorter.fit(HMMSpikeSorter.HMMSpikingModel, template_model, _dataf, 100_000)

    output_data = Dict("mlseq" => modelf.ml_seq,
                              "ll" => modelf.ll,
                              "waveforms" => modelf.template_model.μ,
                              "lp" => lp,
                              "sigma" => modelf.template_model.σ)
    if dosave
        MAT.matwrite(outputfile, output_data)
    end
    print("Done! Results saved to $(outputfile)\n")
    return output_data
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    #pargs = parseargs()
    pargs = my_parse_args(ARGS)
    if !isempty(pargs)
        if !isfile(pargs["inputfile"]) || !isfile(pargs["datafile"])
            print("Both inputfile and data file must exist\n")
            return 23
        else
            sort_data(pargs["inputfile"], pargs["datafile"], pargs["outputfile"]);
        end
    end
    return 0
end

    #pargs = parseargs()
    #sort_data(pargs["inputfile"], pargs["datafile"], pargs["outputfile"])
