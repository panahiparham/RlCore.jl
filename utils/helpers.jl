using DecisionMakingUtils: ZeroOneNormalization, BufferedFunction, FourierBasis, FourierBasisBuffer
using Flux: Chain

using DecisionMakingEnvironments
using BenchmarkEnvironments
using DecisionMakingUtils
using DecisionMakingPolicies
using Plots

import Base: push!



"""
    function make_basis(env, order)
Creates a Fourier basis upto a given order of (coupled and uncoupled) terms
"""
function make_basis(env, order)
    X = env.X  # Matrix Nx2 where N is the number of states and first column is minimum value last is maximum
    num_observations = size(X, 1)
    dorder = order  # cos(pi*(dorder * x1 + dorder*x2)))
    iorder = order  # cos(pi*(iorder * x1)))
    nrm = ZeroOneNormalization(X)
    nbuff = zeros(num_observations)
    nf = BufferedFunction(nrm, nbuff)
    fb = FourierBasis(num_observations, dorder, iorder)
    fbuff = FourierBasisBuffer(fb)
    f = BufferedFunction(fb, fbuff)
    ϕ = Chain(nf, f)
    return ϕ, length(fb)
end



function make_basis(order)

    # if no env is passed in, assume pinball observation space
    X = [
           [0.0 1.0]
           [0.0 1.0]
           [-2.0 2.0]
           [-2.0 2.0]
        ]

    num_observations = size(X, 1)
    dorder = order  # cos(pi*(dorder * x1 + dorder*x2)))
    iorder = order  # cos(pi*(iorder * x1)))
    nrm = ZeroOneNormalization(X)
    nbuff = zeros(num_observations)
    nf = BufferedFunction(nrm, nbuff)
    fb = FourierBasis(num_observations, dorder, iorder)
    fbuff = FourierBasisBuffer(fb)
    f = BufferedFunction(fb, fbuff)
    ϕ = Chain(nf, f)

    return ϕ, length(fb)
end





"""
    epsilon_greedy(qvals, ϵ)
This function returns an action given the q values and the ϵ for ϵ-greedy.
"""
function epsilon_greedy(qvals, ϵ)
    if rand() < ϵ
        return rand(1:length(qvals))
    else
        return argmax(qvals)  # NOTE: This assumes all actions are unique. This should be random over the set of maximizing actions
    end
end


"""
    render_episode(env, policy, save_path)
Creates a gif visualization of an episode and saves
it to save_path.
"""
function render_episode(env, policy, file_path)
    s, x = env.d0()
    anim = Animation()
    p = env.render(s)
    frame(anim, p)

    done = false
    while !done
        a = policy(x)
        s, x, r, γ = sample(env, s, a)
        p = env.render(s)
        frame(anim, p)
        done = γ == 0.0
    end

    g = gif(anim, file_path, fps = 30)
    display(g)

end




# this requires sarsa lambda to be imported!
function visualize_policy(env, policy, num_episodes, save_path)
    for ep=1:num_episodes
        render_episode(env, policy, save_path * string(ep) * ".gif")
    end

end




function validate_policy(env, policy, num_episodes)
    returns = []
    successes = []
    lengths = []

    for _ in 1:num_episodes
        s, x = env.d0()
        a = policy(x)
        done = false
        time_step = 1
        success = 0
        G = 0.0
        γt = 1.0
        while !done
            s, x, r, γ = sample(env, s, a)
            G += γt * r
            γt *= γ

            a = policy(x)

            if γ == 0.0
                done = true
                if s[1] <= env.S[1][2]
                    success = 1
                end
            else
                time_step += 1
            end

        end
        push!(returns, G)
        push!(successes, success)
        push!(lengths, time_step)
    end

    return returns, successes, lengths
end


function generate_trajectories(env, policy, num_episodes; main_subgoal = true)
    trajectories = []

    while length(trajectories) < num_episodes
        τ = collect_data(env, policy)

        final_reward = τ.rewards[end]

        if main_subgoal === false
            # change final reward to be low
            τ.rewards[end] = -1
        end

        if final_reward > 0
            push!(trajectories, τ)
        end
    end

    # if save_trajectories == true
    #     isdir("trajectories") || mkdir("trajectories")
    #     traj_path = "trajectories/policy-$pol_number.jld2"
    #     save(traj_path, "trajectories", trajectories)
    # end    
    
    return trajectories
end


function push!(τ::Trajectory{T, TS, TA}, state::TS, action::TA, blogp::T) where {T, TS, TA}
    push!(τ.states, deepcopy(state))
    push!(τ.actions, deepcopy(action))
    push!(τ.blogps, blogp)
end


# this function collects one episode of interactions from the environment
function collect_data(env, policy)
    τ = Trajectory(env)
    s,x = env.d0()  # get initial state and observation
    done = false  # flag for end of episode
    while !done  # until end of episode
        a = policy(x)  # get action from policy
        blogp = 0.0     # log probability of action taken by the behavior policy. Not implemented in this code. 
        push!(τ, s[2], a, blogp) # TODO is it fine not to save the time step of s = (t, x)? this push! dispatch doesnt expect s to be a tuple
        s, x, r, γ = sample(env, s, a)  # take action and observe reward and next state
        push!(τ.rewards, r)
        done = γ == 0.0 # check if episode is over
    end
    finish!(τ)  # mark the episode as completed. 
    return τ
end


function smoothen_run(run, k)
    smooth_run = zeros(length(run))
    for i=1:length(run)
        s = max(1, i-k+1)
        f = i
        smooth_run[i] = mean(run[s:f])
    end
    return smooth_run
end