using JSON


mutable struct ExperimentModel
    agent::String
    problem::String
    description::String
    episode_cutoff::Int
    total_steps::Int
    _d::Dict
    _keys::String
    _path::String
    _num_perms::Union{Int, Nothing}
    _pairs::Union{Nothing,Vector{Tuple{String, Any}}}
    _save_key::Union{String, Nothing}

    function ExperimentModel(d, path)
        agent = d["agent"]
        problem = d["problem"]

        if Sys.iswindows()
            desc = replace(split(path, "\\")[end], ".json"=>"")
        else
            desc = replace(split(path, "/")[end], ".json"=>"")
        end


        episode_cutoff = get(d, "episode_cutoff", -1)
        total_steps = d["total_steps"]
        _save_key = get(d, "save_key", nothing)

        return new(agent, problem, desc, episode_cutoff, total_steps, d, "metaparameters", path, nothing, nothing, _save_key)
    end

end


function load_experiment(path::String)
    d = JSON.parsefile(path)
    d["exp_path"] = path # Get the name of experiments

    exp = ExperimentModel(d, path)

    return exp
end


function get_run(exp::ExperimentModel, idx::Int)
    count = num_permutations(exp)
    return div(idx, count) + 1
end


function num_permutations(exp::ExperimentModel)
    if exp._num_perms !== nothing
        return exp._num_perms
    end

    if exp._pairs === nothing
        sweeps = permutable(exp)
        exp._pairs = flatten_to_key_values(sweeps)
    end

    exp._num_perms = get_count_from_pairs(exp._pairs)
    return exp._num_perms
end



function permutable(exp::ExperimentModel)
    return exp._d[exp._keys]
end


function get_permutation(exp::ExperimentModel, idx::Int)
    if exp._pairs === nothing
        sweeps = permutable(exp)
        exp._pairs = flatten_to_key_values(sweeps)
    end

    permutation = get_permutation_from_pairs(exp._pairs, idx)
    outer = deepcopy(exp._d)
    delete!(outer, exp._keys)
    d = merge(outer, permutation)

    return d
end

function flatten_to_key_values(sweeps::Dict{String, Any})::Vector{Tuple{String, Any}}
    out::Vector{Tuple{String, Any}} = []
    for (key, values) in sweeps
        push!(out, (key, values))
    end
    return out
end

function get_permutation_from_pairs(pairs::Vector{Tuple{String, Any}}, idx::Int)

    perm = Dict{String, Any}()
    accum = 1

    for (key, values) in pairs

        num = length(values)

        if num == 0
            perm[key] = []
            continue
        end

        perm[key] = values[div(idx - 1, accum) % num + 1]
        accum *= num
    end
    return perm
end


function merge(d1::Dict{String, Any}, d2::Dict{String, Any})
    ret = copy(d2)

    for key in keys(d1)
        ret[key] = get(d2, key, d1[key])
    end

    return ret    
end


function get_count_from_pairs(pairs::Vector{Tuple{String, Any}})
    accum = 1
    for (key, values) in pairs
        num = length(values)
        if num == 0
            num = 1
        end
        accum *= num
    end
    return accum
end

# -----------------
# -- saving path --
# -----------------

struct Config
    save_path::String
    log_path::String
    experiment_directory::String

    function Config(config)
        save_path = get(config, "save_path", "results")
        log_path = get(config, "log_path", "logs")
        experiment_directory = get(config, "experiment_directory", "experiments")
        return new(save_path, log_path, experiment_directory)
    end
end

function get_default_config()
    d = JSON.parsefile("config.json")
    return Config(d)
end

function _get_save_key(exp::ExperimentModel, save_key::Union{String, Nothing} = nothing)
    if save_key !== nothing
        return save_key
    end

    if exp._save_key !== nothing
        return exp._save_key
    end

    config = get_default_config()

    return config.save_path
end
    
function get_keys(exp::ExperimentModel)
    
end

function interpolate(s::String, d::Dict{String, Any})
    for m in eachmatch(r"{.*?}", s)
        key = m.match[2:end-1]
        value = d[key]
        s = replace(s, m.match=>value)
    end

    return s
end


function get_experiment_name(exp::ExperimentModel)
    if Sys.iswindows()
        x = split(exp._path, "\\")
    else
        x = split(exp._path, "/")
    end
    i = findall(x->x=="experiments", x)[1]
    return x[i+1]
end
    

function interpolate_save_path(exp::ExperimentModel, key::Union{String, Nothing} = nothing)
    key = _get_save_key(exp)
    d = Dict{String, Any}(
        "name" => get_experiment_name(exp),
        "agent" => exp.agent,
        "description" => exp.description
        )

    return interpolate(key, d)   
end