using JSON
using RLGlue

include("../environments/pinball.jl")

struct PinballProblem
    environment::RLGlue.BaseEnvironment

    random_start::Bool
    start_location::Union{Tuple{Float64, Float64}, Nothing}
    initiation_radius:: Union{Float64, Nothing}    

    random_goal::Bool
    goal_location::Union{Tuple{Float64, Float64}, Nothing}

    discount_factor::Float64


    function PinballProblem(config_file_path)
        @assert isfile(config_file_path) "missing file $config_file_path"
        data = JSON.parsefile(config_file_path)

        start_in_keys = "start_location" in keys(data)
        init_in_keys = "initiation_radius" in keys(data)
        @assert (start_in_keys && !init_in_keys)||(!start_in_keys && init_in_keys) "include only one of {start_location, initiation_radius} in $config_file_path"

        random_start = !start_in_keys && init_in_keys
        random_goal = !("goal_location" in keys(data))

        initiation_radius = get(data, "initiation_radius", nothing)
        start_location = get(data, "start_location", nothing)
        goal_location = get(data, "goal_location", nothing)

        if !random_start
            start_location = Tuple(start_location)
        end

        if !random_goal
            goal_location = Tuple(goal_location)
        end

        @assert "discount_factor" in keys(data) "missing discount_factor in $config_file_path"
        discount_factor = data["discount_factor"]

        
        environment = Pinball(data["environment"], random_start, random_goal, start_location, goal_location, initiation_radius)

        return new(environment, random_start, start_location, initiation_radius, random_goal, goal_location, discount_factor)
    end
end
