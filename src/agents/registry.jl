include("random.jl")
include("TCSarsaLambda.jl")

function get_agent(agent_name, observations, actions, params, seed)


    if agent_name == "random"
        return RandomAgent(actions, seed)
    elseif agent_name == "TCSarsaLambda"
        return TCSarsaLambda(observations, actions, params, seed)
    elseif agent_name == "ESARSA"
        return ESARSA(observations, actions, params, seed)
    else
        error("Agent not found")
    end
end