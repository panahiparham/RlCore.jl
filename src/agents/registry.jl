include("random.jl")
include("TCSarsaLambda.jl")

function get_agent(agent_name, observations, actions, params, seed, gamma)

    if agent_name == "random"
        return RandomAgent(actions, seed)
    elseif agent_name == "TCSarsaLambda"
        return TCSarsaLambda(observations, actions, params, seed, gamma)
    elseif agent_name == "ESARSA"
        return ESARSA(observations, actions, params, seed, gamma)
    else
        error("Agent not found")
    end
end