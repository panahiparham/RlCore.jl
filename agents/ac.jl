using Statistics
include("../utils/helpers.jl")


# Actor Critic Algorithm with Eligibility Traces and Linear Function Approximation
mutable struct ActorCritic{T,TP,TB,TW,TWP}
    ϕ::TB   # basis function
    W::TW   # state-value weights
    π_θ::TP # policy parameterised by θ
    z_w::TW # eligibility trace for the value function
    z_θ::TWP# eligibility trace for the policy
    α_w::T  # step size for the value function
    α_θ::T  # step size for the policy
    γ::T    # algorithm's discount factor, not the environments
    λ::T    # trace decay parameter
    
    function ActorCritic(num_actions, num_features, ϕ, α_w::T, α_θ::T, γ::T, λ::T) where {T}
        W = zeros(num_features)
        z_w = zero(W)
        π_θ = LinearSoftmax(num_features, num_actions)
        θ = π_θ.θ
        z_θ = zero.(θ)
        return new{T,typeof(π_θ), typeof(ϕ), typeof(W), typeof(z_θ)}(ϕ, W, π_θ, z_w, z_θ, α_w, α_θ, γ, λ)
    end
end


"""
trace_update(alg::ActorCritic, feats, ψ, γ)
    This function updates the eligibility trace for Actor & Critic. 
"""
function trace_update(alg::ActorCritic, feats, ψ, γ)
    @. alg.z_w *= γ * alg.γ * alg.λ
    @. alg.z_w += feats
    @. alg.z_θ *= γ * alg.γ * alg.λ
    @. alg.z_θ += ψ[1]
end


"""
next_action(alg::ActorCritic, s′)
    This function returns the next action, the next state value, and the feature vector for the next state.
    Returning the feature vector for the next state is just to avoid recomputing it.
"""
function next_action(alg::ActorCritic, s′)
    ϕ′ = alg.ϕ(s′)
    v′ = dot(alg.W, ϕ′)
    a′, logp, ψ′ = sample_with_trace(alg.π_θ, ϕ′)
    return a′, v′, ϕ′, ψ′
end


"""
weight_update(alg::ActorCritic, δ, parl)
    This function updates the weights given the TD-error δ.
"""
function weight_update(alg::ActorCritic, δ, parl)
    if parl
        @. alg.W += 0.5*alg.α_w * δ * alg.z_w # tried with 0.1 and that works too
    else
        @. alg.W += alg.α_w * δ * alg.z_w
    end
    @. alg.π_θ.θ += alg.α_θ * δ * alg.z_θ
end


"""
parl2!(alg::ActorCritic, ϕ, ϕ′, γ , a)
    PARL2 adaptive stepsize algorithm for state value functions. 
"""
function parl2!(alg::ActorCritic, ϕ, ϕ′, γ, a)
    Δϕ = γ * alg.γ * ϕ′ - ϕ
    prod = alg.z_w' * Δϕ
    if prod < 0
        alg.α_w = min(alg.α_w, -1/prod)
    end
end


"""
alg_update(alg::ActorCritic, s, a, r, s′, γ, parl)
    This function does computation of the TD error, updates the weights, and computes the next action.
    Note that γ is the discount factor from the environment.
"""
function alg_update(alg::ActorCritic, s, a, r, s′, γ, parl)
    ϕ = alg.ϕ(s)
    v = dot(alg.W, ϕ)
    if γ == 0.0
        a′ = 0.0
        v′ = 0.0
        ϕ′ = ϕ
    else
        a′, v′, ϕ′, ψ′ = next_action(alg, s′)
    end
    δ = r + γ * alg.γ * v′ - v
    weight_update(alg, δ, parl)
    if γ != 0.0
        trace_update(alg, ϕ, ψ′, γ)
        if parl
            parl2!(alg, ϕ, ϕ′,γ, a′)
        end
    end
    return a′
end


"""
begin_episode(alg::ActorCritic, s)
    This initializes the eligibility trace and returns the first action.
"""
function begin_episode(alg::ActorCritic, s)
    fill!(alg.z_w, 0.0)
    fill!(alg.z_θ, 0.0)
    ϕ = alg.ϕ(s)
    a, logp, ψ = sample_with_trace(alg.π_θ, ϕ)
    return a
end


"""
reset_agent!(alg::ActorCritic, num_features, parl)
    This function resets the actor's and critic's weights and eligibility traces
    between runs.
"""
function reset_agent!(alg::ActorCritic, num_features, parl)
    fill!(alg.W, 0.0)
    fill!(alg.z_w, 0.0)
    fill!(alg.z_θ, 0.0)
    fill!(alg.π_θ.θ, 0.0)
    # initialise α_w differently based on whether we're using step size adaptation
    parl ? alg.α_w = 1.0 / num_features : alg.α_w = 0.1 / num_features
end


"""
control_validate(alg::ActorCritic, env, min_episodes, max_episodes, success_threshold, validate_every_n_episodes, validate_for_n_episodes; render_episode=false)
    Validation on the algorithm
"""
function control_validate(alg::ActorCritic, env, min_episodes, max_episodes, success_threshold, validate_every_n_episodes, validate_for_n_episodes)
    train_returns = []
    train_successes = []
    train_lengths = []
    validation_success_rates = []
    validation_mean_lengths = []

    if success_threshold === nothing
        success_threshold = 2
    end

    for episode in 1:max_episodes
        G, success, episode_len = control_episode(alg, env)

        append!(train_returns, G)
        append!(train_successes, success)
        append!(train_lengths, episode_len)

        # validation
        if episode % validate_every_n_episodes == 0
            policy = create_policy(alg)
            _, successes, lengths = validate_policy(env, policy, validate_for_n_episodes)
            success_rate = mean(successes)
            println("Episode: $episode, Validation Success Rate: $success_rate")

            append!(validation_success_rates, success_rate)
            append!(validation_mean_lengths, mean(lengths))
    
            if success_rate >= success_threshold && episode >= min_episodes
                break
            end

        end

    end

    results = Dict(
        "train_returns" => train_returns,
        "train_successes" => train_successes,
        "train_lengths" => train_lengths,
        "validation_success_rates" => validation_success_rates,
        "validation_mean_lengths" => validation_mean_lengths,
    )

    return results
end


"""
control_episode(alg::ActorCritic, env)
    ! Assumes default use of PARL2
    Performs one episode with Actor Critic and update the internal state of alg
    returns: G, success, time_step
"""
function control_episode(alg::ActorCritic, env)
    success = 0
    s, x = env.d0()
    xold = zero(x)

    G = 0.0
    γt = 1.0
    a = begin_episode(alg, x)
    done = false

    time_step = 0
    while !done

        time_step += 1

        @. xold = x
        s, x, r, γ = sample(env, s, a)
        G += γt * r
        γt *= γ
        a′ = alg_update(alg, xold, a, r, x, γ, true)
        a = a′
        if γ == 0.0
            done = true
            if s[1] <= env.S[1][2]
                success = 1
            end
        end
    end

    return G, success, time_step
end


"""
create_policy(alg::ActorCritic)
    This function creates a policy function that selects actions based on the
    learnt distribution over action space.
"""
function create_policy(alg::ActorCritic)
    function policy(s, ϕ, π_θ)
        feats = ϕ(s)
        a, logp, ψ = sample_with_trace(π_θ, feats)
        return a
    end
    ϕ = alg.ϕ
    π_θ = alg.π_θ
    return s->policy(s, ϕ, π_θ) #this is why policy takes in a state!
end

function create_value_function(alg::ActorCritic)
    function value_function(s, ϕ, W)
        feats = ϕ(s)
        return dot(W, feats)
    end
    ϕ = alg.ϕ
    W = alg.W
    return s->value_function(s, ϕ, W)
end