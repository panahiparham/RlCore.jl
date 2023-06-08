include("../agents/sarsa_lambda.jl")
include("../utils/helpers.jl")


function recover_agent_from_weights(w, cfg=Dict())
    # use default values if not provided
    α = get(cfg, "α", 1e-3)
    γ = get(cfg, "γ", 0.99)
    λ = get(cfg, "λ", 0.9)
    ϵ = get(cfg, "ϵ", 0.05)

    # assume pinball env
    num_actions = 5
    ϕ, num_features = make_basis(5)

    println(α, " ", γ, " ", λ, " ", ϵ)
    println(num_actions, " ", num_features)

    agent = LinearSarsa(num_actions, num_features, ϕ, α, γ, λ, ϵ)
    agent.W = w


    return agent
end


