include("../experiment/ExperimentModel.jl")



function gather_missing_indices(exp_paths::Vector{String}, runs::Int, sampler::String)
    path_to_indices = Dict{String, Vector{Int}}()


    for path in exp_paths
        exp = load_experiment(path)
        indices = detect_missing_indices(exp, runs, sampler)
        indices = sort(indices)
        path_to_indices[path] = indices
        _size = num_permutations(exp) * runs
        println("$path $(size(indices)[1]) / $_size")
    end

    return path_to_indices
end




function detect_missing_indices(exp::ExperimentModel, runs::Int, sampler::String)
    all_indices = Vector([x for x in 1:num_permutations(exp) * runs])

    # case 1: no results folder -> all indices are missing
    path_to_results = interpolate_save_path(exp)
    if !isdir(path_to_results)
        return all_indices
    end


    # case 2: no sampler folder -> all indices are missing
    sampler_results_path = joinpath(path_to_results, sampler)
    if !isdir(sampler_results_path)
        return all_indices
    end


    # case 3: no indices -> all indices are missing
    indices = readdir(sampler_results_path) .|> x -> parse(Int, x)

    if isempty(indices)
        return all_indices
    end


    # case 4: some indices -> some indices are missing
    missing_indices = setdiff(all_indices, indices)

    if !isempty(missing_indices)
        return missing_indices
    end

    
    # case 5: all indices -> no indices are missing
    return []

end