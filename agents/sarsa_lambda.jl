using ProgressBars

include("../utils/helpers.jl")
include("../utils/scheduler.jl")


# Linear Sarsa(λ) Algorithm
mutable struct LinearSarsa{T,TB,TW}
    ϕ::TB   # basis function
    W::TW   # q weights
    z::TW   # eligibility trace
    α::T    # this is the step size
    γ::T    # this is the algorithm's discount factor not the environments
    λ::T    # this is the trace decay parameter
    ϵ::T    # this is the ϵ for ϵ-greedy
    α_0::T  # this is the initial step size
    β::T    # this is the emphasis parameter for GSP
    projection::Any # this is the projection function for GSP
    meta_α::T # parl2 step size scale factor

    
    function LinearSarsa(num_actions, num_features, ϕ, α::T, γ::T, λ::T, ϵ::T; β = 0.0, projection = nothing, meta_α = 0.1) where {T}
        W = zeros(num_features, num_actions)
        z = zero(W)
        α_0 = α
        return new{T,typeof(ϕ),typeof(W)}(ϕ, W, z, α, γ, λ, ϵ, α_0, β, projection, meta_α)
    end
end



"""
trace_update(alg::LinearSarsa, feats, a, γ)
    This function updates the eligibility trace for Sarsa(λ). 
"""
function trace_update(alg::LinearSarsa, feats, a, γ)
    @. alg.z *= γ * alg.γ * alg.λ
    @. alg.z[:, a] += feats
end


"""
next_action(alg::LinearSarsa, s′)
    This function returns the next action, the next q value, and the feature vector for the next state.
    Returning the feature vector for the next state is just to avoid recomputing it.
"""
function next_action(alg::LinearSarsa, s′)
    ϕ′ = alg.ϕ(s′)
    q′ = zeros(size(alg.W, 2))
    mul!(q′, alg.W', ϕ′)
    a′ = epsilon_greedy(q′, alg.ϵ)
    return a′, q′[a′], ϕ′
end


"""
weight_update(alg::LinearSarsa, δ, parl)
    This function updates the weights given the TD-error δ.
"""
function weight_update(alg::LinearSarsa, δ, parl)
    if parl
        @. alg.W += alg.meta_α*alg.α * δ * alg.z
    else
        @. alg.W += alg.α * δ * alg.z
    end
end


"""
parl2!(alg::LinearSarsa, ϕ, ϕ′, γ, a)
    PARL2 adaptive stepsize algorithm. 
"""
function parl2!(alg::LinearSarsa, ϕ, ϕ′, γ, a::Int, a′::Int)
    Δ = zeros(size(alg.W))
    Δ[:, a] =  -ϕ
    if a′ > 0
        Δ[:, a′] += γ * alg.γ * ϕ′
    end
    # Δϕ = γ * alg.γ * ϕ′ - ϕ
    # prod = alg.z[:,a]' * Δϕ
    prod = dot(vec(alg.z), vec(Δ))
    if prod < 0
        # print("choose between: ", alg.α_w, -1/prod, "\n")
        alg.α = min(alg.α, -1/prod)
    end

    # println(alg.α)
end


"""
alg_update(alg::LinearSarsa, s, a, r, s′, γ, parl)
    This function does computation of the TD error, updates the weights, and computes the next action. 
    Note that γ is the discount factor from the environment. 
"""
function alg_update(alg::LinearSarsa, s, a, r, s′, γ, parl)
    ϕ = alg.ϕ(s)
    Wa = @view alg.W[:, a]
    q = dot(Wa, ϕ)    
    if γ == 0.0  # terminal state
        a′ = 0
        q′ = 0.0
        ϕ′ = ϕ
    else
        a′, q′, ϕ′ = next_action(alg, s′)
    end

    if alg.projection === nothing
        δ = r + γ * alg.γ * q′ - q
    else
        v_sub, outside_initation = alg.projection(s, a)

        if outside_initation
            δ = r + γ * alg.γ * q′ - q
        else
            δ = r + γ * alg.γ * ((1-alg.β)*q′ + alg.β*v_sub) - q
        end
    end

    # if outside_initation
    #     δ = r + γ * alg.γ * q′ - q
    # else
    #     δ = r + γ * alg.γ * ((1-β)*q′ + β*v_sub) - q
    # end
    if parl
        # println(a, a′)
        parl2!(alg, ϕ, ϕ′, γ, a, a′) # if we wish to use parl2 optimizer on stepsize.
    end
    weight_update(alg, δ, parl)
    if γ != 0.0
        trace_update(alg, ϕ′, a′, γ) 
    end
    return a′
end


"""
begin_episode(alg::LinearSarsa, s)
    This initializes the eligibility trace and returns the first action.
"""
function begin_episode(alg::LinearSarsa, s)
    fill!(alg.z, 0.0)
    ϕ = alg.ϕ(s)
    q = zeros(size(alg.W, 2))
    mul!(q, alg.W', ϕ)
    a = epsilon_greedy(q, alg.ϵ)
    trace_update(alg, ϕ, a, 0.0)
    return a
end


"""
reset_agent!(alg::LinearSarsa)
    This function resets the agent's weights and eligibility trace
    between runs.
"""
function reset_agent!(alg::LinearSarsa)
    alg.W = zeros(size(alg.W))
    alg.z = zeros(size(alg.z))
    alg.α = alg.α_0
end


## TASK RUNNING

"""
control_validate(alg::LinearSarsa, env, min_episodes, max_episodes, success_threshold, validate_every_n_episodes, validate_for_n_episodes)
    Validation on the algorithm
"""
function control_validate(alg::LinearSarsa, env, min_episodes, max_episodes, success_threshold, validate_every_n_episodes, validate_for_n_episodes)


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
function control_episode(alg::LinearSarsa, env)

Run one episode of GSP-Sarsa(λ) on the environment and update the internal state of alg.
If β and projection are not provided then this is just Sarsa(λ). 
This function uses PARL2 for the stepsize adaptation.

inputs
    - alg: LinearSarsa struct
    - env: environment
    - parl::Bool use Parl2? (default true)

outputs
    - G: return of the episode
    - success: 1 if the episode ends in reaching the goal 0 otherwise
    - time_step: length of the episode
"""
function control_episode(alg::LinearSarsa, env; parl=true)
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
        a′ = alg_update(alg, xold, a, r, x, γ, parl)
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
function episodic_control(alg::LinearSarsa, env, num_episodes)

controls num_episodes episodes of the agent on the environment

inputs
    - alg: LinearSarsa struct
    - env: environment
    - num_episodes: number of episodes to run
    - reset_agent: if true then reset the agent at the start (default true)

    outputs
    - Gs: array of returns - shape: (num_episodes)
    - Ss: array of success - shape: (num_episodes)
    - Ts: array of episode lengths - shape: (num_episodes)
    - weights: array of weights - shape: (num_episodes, size(alg.W))
"""
function episodic_control(alg::LinearSarsa, env, num_episodes; reset_agent = true, step_decay = 1.0, eps_decay = 1.0)
    if reset_agent
        reset_agent!(alg)
    end
    
    # TODO: should we store everything
    Gs, Ss, Ts, weights = zeros(num_episodes), zeros(num_episodes), zeros(num_episodes), zeros(num_episodes, size(alg.W)...)

    # TODO: should these exist?
    step_schedule = scheduler(alg.α, step_decay, 0)
    eps_schedule = scheduler(alg.ϵ, eps_decay, 0)

    for ep in ProgressBar(1:num_episodes)
        G, success, t = control_episode(alg, env)
        Gs[ep] = G
        Ss[ep] = success
        Ts[ep] = t
        weights[ep, :, :] = alg.W

        alg.α = step_schedule()
        alg.ϵ = eps_schedule()

    end

    return Gs, Ss, Ts, weights
end


"""
function multirun_episodic_control(alg::LinearSarsa, env, num_runs, num_episodes)

controls num_episodes episodes of the agent on the environment for num_runs runs.
Resets the agent between runs.

inputs
    - alg: LinearSarsa struct
    - env: environment
    - num_runs: number of runs to do
    - num_episodes: number of episodes to run for each run
outputs
    - Gs: array of returns - shape: (num_runs, num_episodes)
    - Ss: array of success - shape: (num_runs, num_episodes)
    - Ts: array of episode lengths - shape: (num_runs, num_episodes)
    - weights: array of weights - shape: (num_runs, num_episodes, size(alg.W))
"""

function multirun_episodic_control(alg::LinearSarsa, env, num_runs, num_episodes)
    Gs, Ss, Ts, weights = zeros(num_runs, num_episodes), zeros(num_runs, num_episodes), zeros(num_runs, num_episodes), zeros(num_runs, num_episodes, size(alg.W)...)

    for run in 1:num_runs
        Gs[run, :], Ss[run, :], Ts[run, :], weights[run, :, :, :] = episodic_control(alg, env, num_episodes; reset_agent = true)
    end
    return Gs, Ss, Ts, weights
end



## POLICY EXTRACTION

"""
create_policy(alg::LinearSarsa, random=false)
    This function creates a policy function that selects actions from based on the q values. 
    If random is true then the policy is ϵ-greedy. Otherwise it is greedy.
"""
function create_policy(alg::LinearSarsa, random=false)
    function policy(s, q, ϕ, W, ϵ)
        feats = ϕ(s)
        mul!(q, W', feats)
        return epsilon_greedy(q, ϵ)
    end
    ϕ = alg.ϕ
    W = alg.W
    q = zeros(size(W, 2))
    ϵ = random ? alg.ϵ : 0.0
    return s->policy(s, q, ϕ, W, ϵ)
end



function create_policy(weights, random=false; order=5, eps=0.05)
    function policy(s, q, ϕ, W, ϵ)
        feats = ϕ(s)
        mul!(q, W', feats)
        return epsilon_greedy(q, ϵ)
    end

    ϕ, _ = make_basis(order)
    W = weights
    q = zeros(size(W, 2))
    ϵ = random ? eps : 0.0
    return s->policy(s, q, ϕ, W, ϵ)
end


"""
function get_greedy_policy(alg::LinearSarsa)

extract the greedy policy of a Sarsa(λ) agent

inputs
    - alg: LinearSarsa struct
outputs
    - policy: greedy policy of the agent
"""
function get_greedy_policy(alg::LinearSarsa)
    policy = create_policy(alg, false)
    return policy
end

function get_greedy_policy(w; order=5)
    policy = create_policy(w, false; order=order)
    return policy
end


"""
function get_e_greedy_policy(alg::LinearSarsa)

extract the e-greedy policy of a Sarsa(λ) agent

inputs
    - alg: LinearSarsa struct
outputs
    - policy: e-greedy policy of the agent
"""
function get_e_greedy_policy(alg::LinearSarsa)
    policy = create_policy(alg, true)
    return policy
end

function get_e_greedy_policy(w, eps; order=5)
    policy = create_policy(w, true; order=order, eps=eps)
    return policy
end



## VALUE FUNCTION EXTRACTION

"""
function qval_fn(alg::LinearSarsa)

returns the action value function for a given state action pair

inputs
    - alg: LinearSarsa struct
outputs
    - value_function: function that returns the action value for a given state action pair.
"""
function qval_fn(alg::LinearSarsa)
    function value_function(s, a, q, ϕ, W)
        feats = ϕ(s)
        mul!(q, W', feats)
        qval = q[a]
        return qval
    end
    ϕ = alg.ϕ
    W = alg.W
    q = zeros(size(W, 2))
    return (s,a) -> value_function(s, a, q, ϕ, W) 
end

function qval_fn(weights; order=5)
    function value_function(s, a, q, ϕ, W)
        feats = ϕ(s)
        mul!(q, W', feats)
        qval = q[a]
        return qval
    end
    ϕ, _ = make_basis(order)
    W = weights
    q = zeros(size(W, 2))
    return (s,a) -> value_function(s, a, q, ϕ, W) 
end


"""
function greedy_qval_fn(alg::LinearSarsa)
    
returns the highest (aka greedy) action value of a given state.

inputs
    - alg: LinearSarsa struct
outputs
    - value_function: function that returns the highest action value of a given state.
"""
function greedy_qval_fn(alg::LinearSarsa)
    function value_function(s, q, ϕ, W)
        feats = ϕ(s)
        mul!(q, W', feats)
        return maximum(q)
    end
    ϕ = alg.ϕ
    W = alg.W
    q = zeros(size(W, 2))
    return s->value_function(s, q, ϕ, W)
end

function greedy_qval_fn(weights; order=5)
    function value_function(s, q, ϕ, W)
        feats = ϕ(s)
        mul!(q, W', feats)
        return maximum(q)
    end
    ϕ, _ = make_basis(order)
    W = weights
    q = zeros(size(W, 2))
    return s->value_function(s, q, ϕ, W)
end



"""
function sval_fn(agent::LinearSarsa)

extract the state value function for the e-greedy policy of a LinearSarsa agent

inputs
    - alg: LinearSarsa struct
outputs
    - value_fn: state value function for the e-greedy policy of the agent
"""
function sval_fn(alg::LinearSarsa)
    function value_func(s, q, ϕ, W, ϵ)
        feats = ϕ(s)
        mul!(q, W', feats)
        value = 0.0
        greedy_action = argmax(q)

        for a=eachindex(q)
            if a == greedy_action
                value += q[a] * ((1 - ϵ) + ϵ/length(q))
            else
                value += q[a] * ϵ/length(q)
            end
        end

        return value
    end

    ϕ = alg.ϕ
    W = alg.W
    ϵ = alg.ϵ
    q = zeros(size(W, 2))

    value_fn = s -> value_func(s, q, ϕ, W, ϵ)

    return value_fn
end

function sval_fn(weights, eps; order=5)
    function value_func(s, q, ϕ, W, ϵ)
        feats = ϕ(s)
        mul!(q, W', feats)
        value = 0.0
        greedy_action = argmax(q)

        for a=eachindex(q)
            if a == greedy_action
                value += q[a] * ((1 - ϵ) + ϵ/length(q))
            else
                value += q[a] * ϵ/length(q)
            end
        end

        return value
    end

    ϕ, _ = make_basis(order)
    W = weights
    ϵ = eps
    q = zeros(size(W, 2))

    value_fn = s -> value_func(s, q, ϕ, W, ϵ)

    return value_fn
end