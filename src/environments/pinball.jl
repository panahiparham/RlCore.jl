using RLGlue
using DecisionMakingEnvironments

include("pinball_custom.jl")


"""
    Pinball environment
"""
mutable struct Pinball <: RLGlue.BaseEnvironment
    mdp::DecisionMakingEnvironments.SequentialProblem
    observations::Array
    actions::Int
    s::Tuple{Float64, Vector{Float64}}

    function Pinball(env_name, random_start, random_goal, start_location, goal_location, initiation_radius)

        if random_start && random_goal
            mdp, _ = custom_pinball(env_name*".cfg", initiation_radius; maxT = 100_000_000_000)
        elseif random_start
            mdp, _ = custom_pinball(env_name*".cfg", initiation_radius; maxT = 100_000_000_000, subgoal_locs = [goal_location])
        elseif random_goal
            mdp, _ = custom_pinball(env_name*".cfg", 2.0; fixed_start = start_location, maxT = 100_000_000_000)
        else
            mdp, _ = custom_pinball(env_name*".cfg", 2.0; fixed_start = start_location, maxT = 100_000_000_000, subgoal_locs = [goal_location])
        end

        raw_observations = mdp.S[2]

        observations::Array{Tuple{Float64, Float64}} = []
        for i=axes(raw_observations, 1)
            push!(observations, (raw_observations[i, 1], raw_observations[i, 2]))
        end
        actions = length(mdp.A)

        s = (0.0, [0.0, 0.0, 0.0, 0.0])
        return new(mdp, observations, actions, s)
    end
end

function render(env::Pinball)
    return env.mdp.render(env.s)
end

function RLGlue.start!(env::Pinball)
    env.s, _ = env.mdp.d0()
    return env.s[2]
end

function RLGlue.step!(env::Pinball, action::Int)
    @assert action in 1:env.actions "Invalid action: $action"

    env.s, o, r, γ = sample(env.mdp, env.s, action)
    println(action, " ", o, " ", r, " ", γ)
    sleep(1)
    if γ == 0.0
        t = true
    else
        t = false
    end

    # modified reward function
    r = -1.0

    return (r, o, t, Dict{String, Any}())
end