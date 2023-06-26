using CSV, Tables, DelimitedFiles
using Statistics

include("collector.jl")
include("../experiment/ExperimentModel.jl")


function save_collector(collector::Collector, exp::ExperimentModel)
    for (name, data) in collector._name_idx_data
        for (idx, values) in data
            path = joinpath(interpolate_save_path(exp), name, string(idx))
            mkpath(path)
            t = Tables.table(values)
            CSV.write(joinpath(path, "data.csv"),  Tables.table(values), writeheader=false)
        end
    end
end

mutable struct ResultsData
    _exp::ExperimentModel
    _name_idx_data::Dict{String, Dict{Int, Array}}

    function ResultsData(exp, name_idx_data)
        return new(exp, name_idx_data)
    end
end

function get_data(path_to_results::String)
    _name_idx_data = Dict{String, Dict{Int, Array}}()

    # get samplers
    samplers = readdir(path_to_results)
    for sampler in samplers
        _name_idx_data[sampler] = Dict{Int, Array}()

        # get indices
        indices = readdir(joinpath(path_to_results, sampler))
        for idx in indices
            # get data
            _path = joinpath(path_to_results, sampler, idx, "data.csv") 
            _name_idx_data[sampler][parse(Int, idx)] = readdlm(_path, ',', Float64)[:, 1]
        end
    end

    return _name_idx_data
end


function load_results(description::String)
    exp = load_experiment(description)
    name_idx_data = get_data(interpolate_save_path(exp))
    
    return ResultsData(exp, name_idx_data)
end


function merge_runs!(_d::ResultsData)
    # The objective of this function is to pool different runs with the same parameter setting
    # So indtead of having indices 0 to num_runs x num_permutations each with shape (num_steps,)
    # we have indices 0 to num_permutations each with shape (num_steps, num_runs)

    # get num_permutations
    num_perms = num_permutations(_d._exp)


    for sampler in keys(_d._name_idx_data)
        # construct new dict
        new_d = Dict{Int, Array}()

        for idx in keys(_d._name_idx_data[sampler])
            # get data
            data = _d._name_idx_data[sampler][idx]

            # get collapsed idx
            c_idx = (idx - 1) % num_perms + 1

            # add to new dict
            if haskey(new_d, c_idx)
                new_d[c_idx] = hcat(new_d[c_idx], data)
            else
                new_d[c_idx] = data
            end
        end

        # replace old dict
        _d._name_idx_data[sampler] = new_d
    end
end


function pick_best_idx(_d::ResultsData, sampler::String, pereferences = "hi", summarizer = mean)
    # The objective of this function is to return the index of the parameter
    # This works for both collapsed and rolled out indices.

    if !haskey(_d._name_idx_data, sampler)
        @error "Sampler not found"
    end

    # get data
    data = _d._name_idx_data[sampler]

    # summerization buffer
    s = Dict{Int, Float64}()

    for idx in keys(data)
        s[idx] = summarizer(data[idx])
    end

    # get best idx
    if pereferences == "hi"
        best_idx = findmax(s)[2]
    elseif pereferences == "lo"
        best_idx = findmin(s)[2]
    else
        @error "Preferences not recognized"
    end

    return best_idx
end


function std_summatizer(data::Array)
    line = mean(data)
    lo = line - std(data)
    hi = line + std(data)
        
    return lo, line, hi
end

function stderr_summarizer(data::Array)
    line = mean(data)
    lo = line - std(data) / sqrt(size(data)[1])
    hi = line + std(data) / sqrt(size(data)[1])
        
    return lo, line, hi
end


function get_learning_curve(_d::ResultsData, sampler::String, idx::Int, summarizer = stderr_summarizer)
    # This function returns a tuple of (lo, line, hi) for plotting learning curves
    # each of lo, line, hi are of shape (num_steps,)
    # the shaded region is computed via applying the summarizer to the runs at each step


    # get num_steps
    data = _d._name_idx_data[sampler][idx]
    num_steps = size(data)[1]
    
    lo = zeros(num_steps)
    line = zeros(num_steps)
    hi = zeros(num_steps)

    for step in 1:num_steps
        lo[step], line[step], hi[step] = summarizer(data[step, :])
    end

    return lo, line, hi
end