include("random.jl")

function get_agent(agent_name, num_actions)

    if agent_name == "random"
        return RandomAgent(num_actions)
    else
        error("Agent not found")
    end
    
end