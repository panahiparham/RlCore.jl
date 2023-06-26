using RLGlue
using TileCoder

include("../utils/argmax.jl")


mutable struct TCSarsaLambda <: RLGlue.BaseAgent

    alpha
    lambda
    w
    z
    last_x
    last_action

    observations
    actions
    params
    seed
    rng
    rep
    gamma
    epsilon

    function TCSarsaLambda(observations::Array, actions::Int, params::Dict, seed::Int, gamma::Float64)

        @assert haskey(params, "alpha") "Missing alpha parameter"
        @assert haskey(params, "lambda") "Missing lambda parameter"
        @assert haskey(params, "tiles") "Missing tiles parameter"
        @assert haskey(params, "tilings") "Missing tilings parameter"



        rng = MersenneTwister(seed);
        epsilon = get(params, "epsilon", 0.0)

        tc_config = TileCoderConfig(params["tiles"], params["tilings"], length(observations); offset = "cascade", scale_output = false, input_ranges = observations, bound = "clip")
        rep = TC(tc_config, rng)

        # initial weights
        w = zeros(actions, features(rep))
        z = zeros(actions, features(rep))

        last_x = nothing
        last_action = nothing


        new(params["alpha"], params["lambda"], w, z, last_x, last_action, observations, actions, params, seed, rng, rep, gamma, epsilon)
    end
end





##### functions #####

function policy(agent::TCSarsaLambda, x::Array)
    q = _values(agent, x)

    if rand(agent.rng) < agent.epsilon
        a = rand(agent.rng, 1:agent.actions)
    else
        a = _argmax(q)
    end
    return a
end

function _values(agent::TCSarsaLambda, x::Array)
    return dropdims(sum(agent.w[:, x], dims=2), dims=2)
end

function update(agent::TCSarsaLambda, x, r)
    last_x = agent.last_x
    
    q = _values(agent, last_x)
    a = agent.last_action

    if x === nothing    # terminal state
        δ = r - q[a]    # TD error
        agent.w += ((agent.alpha / agent.params["tilings"]) * δ) .* agent.z # weight update


    else
        q′ = _values(agent, x)
        a′ = policy(agent, x)
        
        δ = r + agent.gamma * q′[a′] - q[a] # TD error

        # maybe bug here?

        agent.w += ((agent.alpha / agent.params["tilings"]) * δ) .* agent.z # weight update
        agent.z .*= agent.gamma * agent.lambda    # trace decay
        agent.z[a′, x] .= 1.0   # replacing trace 
    end
end

function cleanup(agent::TCSarsaLambda)
    return nothing
end


##### RLGlue interface #####

function RLGlue.start!(agent::TCSarsaLambda, observation::Any)
    x = get_indices(agent.rep, observation)
    a = policy(agent, x) 

    agent.last_x = x
    agent.last_action = a
    return a
end

function RLGlue.step!(agent::TCSarsaLambda, reward::Float64, observation::Any, extra::Dict{String, Any})
    x = get_indices(agent.rep, observation)

    update(agent, x, reward)
    
    a = policy(agent, x)

    agent.last_x = x
    agent.last_action = a
    return a
end

function RLGlue.end!(agent::TCSarsaLambda, reward::Float64, extra::Dict{String, Any})
    update(agent, nothing, reward)
    cleanup(agent)
    return nothing
end