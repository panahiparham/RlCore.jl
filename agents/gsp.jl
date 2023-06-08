using JLD2

include("random_agent.jl")
include("sarsa_lambda.jl")
include("gsp_sarsa_lambda.jl")
include("ac.jl")
include("regression.jl")
include("../environments/pinball_custom.jl")
include("../utils/helpers.jl")


mutable struct gsp{T, TE, TA, TB}
    env::TE                                     # environment
    agt::TA                                     # agent TODO check if we need this
    num_features::T                             # number of features
    num_actions::T                              # number of actions
    ϕ::TB                                       # basis function
    option_policies::Array{Function, 1}         # option policy functions, π_g(a|s)
    s2g_reward_models::Array{Function, 1}       # state to subgoal reward models, r(s,g)
    s2g_discount_models::Array{Function, 1}     # state to subgoal discount models, Γ(s,g)
    τs::Array{Array{Trajectory, 1}, 1}          # list of lists trajectories leading to each subgoal (from s2g, for g2g)
    g2g_reward_model::Matrix{Float64}           # subgoal to subgoal reward model, r(g,g')
    g2g_discount_model::Matrix{Float64}         # subgoal to subgoal discount model, Γ(g,g')
    v_AMDP::Array{Float64}                      # tabular value function for the AMDP
    meta_data::Dict                             # experiment configuration data

    function gsp(meta_data)
        env_config_file = meta_data["env_config_file"]
        initiation_radius = meta_data["initiation_radius"]
        subgoal_locations = meta_data["subgoal_locations"]
        start_location = meta_data["start_location"]
        max_time_steps = meta_data["max_time_steps"]
        feature_order = meta_data["feature_order"]

        env, _  = custom_pinball(env_config_file, initiation_radius; maxT = max_time_steps, subgoal_locs=subgoal_locations, fixed_start = start_location)
        ϕ, num_features = make_basis(env, feature_order)
        num_actions = length(env.A)

        agt = (projection, β) -> GSPLinearSarsa(num_actions, num_features, ϕ, 0.1, 0.99, 0.9, 0.05, β, projection) 

        # Setup option policies
        option_policies = Array{Function, 1}(undef, length(subgoal_locations))

        # Setup s2g reward and discount models
        s2g_reward_models = Array{Function, 1}(undef, length(subgoal_locations))
        s2g_discount_models = Array{Function, 1}(undef, length(subgoal_locations))

        # Setup training trajectories
        trajectories = Array{Trajectory,1}[]

        # Setup g2g reward and discount models
        g2g_reward_model = zeros(length(subgoal_locations), length(subgoal_locations))
        g2g_discount_model = zeros(length(subgoal_locations), length(subgoal_locations))

        v_AMDP = zeros(length(subgoal_locations))

        return new{typeof(num_features), typeof(env), typeof(agt), typeof(ϕ)}(
                env, agt, num_features, num_actions, ϕ, option_policies, s2g_reward_models,
                s2g_discount_models, trajectories, g2g_reward_model, g2g_discount_model, v_AMDP, meta_data)
    end
end


# learn option policies
function learn_option_policies(alg::gsp)

    # reset learned option policies
    alg.option_policies = Array{Function, 1}(undef, length(alg.meta_data["subgoal_locations"]))

    min_episodes = alg.meta_data["option_policy_learning"]["min_episodes"]
    max_episodes = alg.meta_data["option_policy_learning"]["max_episodes"]
    max_steps = alg.meta_data["option_policy_learning"]["max_steps"]
    success_threshold = alg.meta_data["option_policy_learning"]["success_threshold"]
    learning_alg = alg.meta_data["option_policy_learner"]

    println("Learning option policies using $learning_alg")

    if learning_alg == "sarsa"
        α′ = alg.meta_data["option_policy_learning"][learning_alg]["α"]
        γ = alg.meta_data["option_policy_learning"][learning_alg]["γ"]
        λ = alg.meta_data["option_policy_learning"][learning_alg]["λ"]
        ϵ = alg.meta_data["option_policy_learning"][learning_alg]["ϵ"]
    elseif learning_alg == "actor_critic"
        α_w′ = alg.meta_data["option_policy_learning"][learning_alg]["α_w"]
        α_θ′ = alg.meta_data["option_policy_learning"][learning_alg]["α_θ"]
        γ = alg.meta_data["option_policy_learning"][learning_alg]["γ"]
        λ = alg.meta_data["option_policy_learning"][learning_alg]["λ"]
    else
        error("Invalid option policy learner")
    end

    validate_every_n_episodes = alg.meta_data["option_policy_learning"]["validate_every_n_episodes"]
    validate_for_n_episodes = alg.meta_data["option_policy_learning"]["validate_for_n_episodes"]

    features_order = alg.meta_data["option_policy_learning"]["features_order"]


    println("min_episodes: ", min_episodes, " max_episodes: ", max_episodes, " max_steps: ", max_steps, " success_threshold: ", success_threshold)

    println("Subgoals: ")
    results = []

    for (i, subgoal) in enumerate(alg.meta_data["subgoal_locations"])
        println(subgoal)
        # learn option policy
        option_policy_env, _  = custom_pinball(alg.meta_data["env_config_file"], alg.meta_data["initiation_radius"]; maxT = max_steps, subgoal_locs=[subgoal], fixed_start = nothing)
        ϕ, num_features = make_basis(option_policy_env, features_order)

        if learning_alg == "sarsa" # Using Sarsa(λ)
            α = α′ / num_features
            num_actions = length(option_policy_env.A)
            option_policy_learner = LinearSarsa(num_actions, num_features, ϕ, α, γ, λ, ϵ)
        elseif learning_alg == "actor_critic" # Using Actor Critic
            α_w = α_w′ / num_features
            α_θ = α_θ′ / num_features
            num_actions = length(option_policy_env.A)
            option_policy_learner = ActorCritic(num_actions, num_features, ϕ, α_w, α_θ, γ, λ)
        else
            error("Invalid option policy learner")
        end
        
        result = control_validate(option_policy_learner, option_policy_env, min_episodes, max_episodes, success_threshold, validate_every_n_episodes, validate_for_n_episodes)

        # animate option policy
        π_g = create_policy(option_policy_learner)
        # render_episode(option_policy_env, π_g, "temp_stuff/$(learning_alg)_opt_pol_$(i).gif")

        # TURN THIS ON FOR HEATMAPS
        # temp for heatmap test
        # v = create_value_function(option_policy_learner)
        # return v
        
        # TODO: print validation stuff

        # println("Validation results: $(results)")
        # p = plot(results["validation_mean_lengths"], label="mean lengths")
        # display(p)

        # save option policy
        alg.option_policies[i] = π_g
        results = [results; result]
    end

    # save validation results
    println("saving results")
    save("temp_stuff/$(learning_alg)_val.jld2", "val_res", results)

    return results
end


# learn s2g models
function learn_s2g_models(alg::gsp, learner="SGD")

    # reset learned s2g models
    alg.s2g_reward_models = Array{Function, 1}(undef, length(alg.meta_data["subgoal_locations"]))
    alg.s2g_discount_models = Array{Function, 1}(undef, length(alg.meta_data["subgoal_locations"]))
    alg.τs = Array{Trajectory,1}[]

    γ = alg.meta_data["γ"]
    init_radius = alg.meta_data["initiation_radius"]
    num_train_trajectories = alg.meta_data["s2g_model_learning"]["num_train_trajectories"]
    num_test_trajectories = alg.meta_data["s2g_model_learning"]["num_test_trajectories"]


    epochs = alg.meta_data["s2g_model_learning"]["epochs"]
    batch_size = alg.meta_data["s2g_model_learning"]["batch_size"]
    learning_rate = alg.meta_data["s2g_model_learning"]["learning_rate"]


    println("Learning s2g models")

    results = []

    for (i,subgoal) in enumerate(alg.meta_data["subgoal_locations"])

        option_policy_env, _  = custom_pinball(alg.meta_data["env_config_file"], alg.meta_data["initiation_radius"]; maxT = alg.meta_data["max_time_steps"], subgoal_locs=[subgoal], fixed_start = nothing)
        option_policy = alg.option_policies[i]

        # generate training trajectories
        train_trajectories = generate_trajectories(option_policy_env, option_policy, num_train_trajectories; main_subgoal = i == 1)
        test_trajectories = generate_trajectories(option_policy_env, option_policy, num_test_trajectories; main_subgoal = i == 1)

        # save trajectories leading to this subgoal
        push!(alg.τs, vcat(train_trajectories, test_trajectories))
         
        # generate datasets
        train_inputs, train_reward_model_outputs, train_discount_model_outputs = generate_s2g_datasets(train_trajectories, subgoal, γ, init_radius)
        test_inputs, test_reward_model_outputs, test_discount_model_outputs = generate_s2g_datasets(test_trajectories, subgoal, γ, init_radius)

        # println(size(train_inputs), size(train_reward_model_outputs), size(train_discount_model_outputs))
        # println(size(test_inputs), size(test_reward_model_outputs), size(test_discount_model_outputs))

        if learner == "OLS"
            ## LEARN S2G USING ORDINARY LEAST SQUARES REGRESSION ##
            r_model_learner = OLS(alg.num_features, alg.num_actions, alg.ϕ)
            train!(r_model_learner, train_inputs, train_reward_model_outputs)
            println("reward model r2 train score: ", validate(r_model_learner, train_inputs, train_reward_model_outputs))
            println("reward model r2 test score: ", validate(r_model_learner, test_inputs, test_reward_model_outputs))

            Γ_model_learner = OLS(alg.num_features, alg.num_actions, alg.ϕ)   
            train!(Γ_model_learner, train_inputs, train_discount_model_outputs)
            println("discount model r2 train score: ", validate(Γ_model_learner, train_inputs, train_discount_model_outputs))
            println("discount model r2 test score: ", validate(Γ_model_learner, test_inputs, test_discount_model_outputs))

            alg.s2g_reward_models[i] = (s, a) -> predict(r_model_learner, [[s, a, 0]])[1]
            alg.s2g_discount_models[i] = (s, a) -> predict(Γ_model_learner, [[s, a, 0]])[1]

            println(predict(r_model_learner, train_inputs))
            println(train_reward_model_outputs)
        
        elseif learner == "SGD"
            ## LEARN S2G USING SGD REGRESSION ##
            # r_model_learner = Adam_regression(alg.num_features, alg.num_actions, alg.ϕ)
            # Γ_model_learner = Adam_regression(alg.num_features, alg.num_actions, alg.ϕ)
            r_model_learner = SGD_regression(alg.num_features, alg.num_actions, alg.ϕ)
            Γ_model_learner = SGD_regression(alg.num_features, alg.num_actions, alg.ϕ)

            reward_train_losses = []
            discount_train_losses = []
            reward_test_losses = []
            discount_test_losses = []

            reward_train_r2s = []
            discount_train_r2s = []
            reward_test_r2s = []
            discount_test_r2s = []


            for epoch in 1:epochs
                # train
                train(r_model_learner, train_inputs, train_reward_model_outputs, batch_size, learning_rate)
                train(Γ_model_learner, train_inputs, train_discount_model_outputs, batch_size, learning_rate)

                # test

                reward_learner_trainloss = test(r_model_learner, train_inputs, train_reward_model_outputs)
                discount_learner_trainloss = test(Γ_model_learner, train_inputs, train_discount_model_outputs)

                reward_learner_testloss = test(r_model_learner, test_inputs, test_reward_model_outputs)
                discount_learner_testloss = test(Γ_model_learner, test_inputs, test_discount_model_outputs)

                println("Epoch: ", epoch)
                println(" Reward Train loss: ", reward_learner_trainloss, " Reward Test loss: ", reward_learner_testloss)
                println(" Discount Train loss: ", discount_learner_trainloss, " Discount Test loss: ", discount_learner_testloss)
                println()

                push!(reward_train_losses, reward_learner_trainloss)
                push!(discount_train_losses, discount_learner_trainloss)
                push!(reward_test_losses, reward_learner_testloss)
                push!(discount_test_losses, discount_learner_testloss)

                reward_learner_trainr2 = r2score(r_model_learner, train_inputs, train_reward_model_outputs)
                discount_learner_trainr2 = r2score(Γ_model_learner, train_inputs, train_discount_model_outputs)

                reward_learner_testr2 = r2score(r_model_learner, test_inputs, test_reward_model_outputs)
                discount_learner_testr2 = r2score(Γ_model_learner, test_inputs, test_discount_model_outputs)

                println("Epoch: ", epoch)
                println(" Reward Train r2: ", reward_learner_trainr2, " Reward Test r2: ", reward_learner_testr2)
                println(" Discount Train r2: ", discount_learner_trainr2, " Discount Test r2: ", discount_learner_testr2)
                println()

                push!(reward_train_r2s, reward_learner_trainr2)
                push!(discount_train_r2s, discount_learner_trainr2)
                push!(reward_test_r2s, reward_learner_testr2)
                push!(discount_test_r2s, discount_learner_testr2)

            end
            # [1] because we only have one input, so we want the first entry in the array
            alg.s2g_reward_models[i] = (s, a) -> predict(r_model_learner, [[s, a, 0]])[1]
            alg.s2g_discount_models[i] = (s, a) -> predict(Γ_model_learner, [[s, a, 0]])[1]
        end
        results = Dict("reward_train_losses" => reward_train_losses, "discount_train_losses" => discount_train_losses, "reward_test_losses" => reward_test_losses, "discount_test_losses" => discount_test_losses, "reward_train_r2s" => reward_train_r2s, "discount_train_r2s" => discount_train_r2s, "reward_test_r2s" => reward_test_r2s, "discount_test_r2s" => discount_test_r2s)
    end
    return results
end


function generate_s2g_datasets(trajectories, subgoal, γ, init_radius)
    
    inputs = []
    reward_model_outputs = []
    discount_model_outputs = []

    for τ in trajectories
        T = size(τ.rewards, 1)
        reward_cumulant = 0.0
        discount_cumulant = 0.0

        for t in T:-1:1
            r = τ.rewards[t]
            s = τ.states[t]
            a = τ.actions[t]

            reward_cumulant = γ*reward_cumulant + r
        
            if τ.done && t == T
                discount_cumulant = γ*discount_cumulant + 1
            else
                discount_cumulant = γ*discount_cumulant
            end

            # skip data outside of initiation radius
            skip_observation = false

            if (s[1] - subgoal[1])^2 + (s[2] - subgoal[2])^2 > init_radius^2
                skip_observation = true
            end
        
            if !skip_observation
                push!(inputs, (s,a,r))
                push!(reward_model_outputs, reward_cumulant)
                push!(discount_model_outputs, discount_cumulant)
            end

        end

    end

    return inputs, reward_model_outputs, discount_model_outputs
end


# learn g2g models
function learn_g2g_models(alg::gsp)
    println("learning g2g models")
    num_updates = zeros(size(alg.g2g_reward_model))
    subgoal_locations = alg.meta_data["subgoal_locations"]
    subgoal_radius = alg.meta_data["subgoal_radius"]
    init_rad = alg.meta_data["initiation_radius"]

    for (end_goal, trajs) in enumerate(alg.τs)
        for τ in trajs
            for (obs, action) in zip(τ.states, τ.actions)
                in_subgoal, start_goal = check_subgoals(obs, subgoal_locations, subgoal_radius) # checks if m(s,g) = 1, for any g ∈ 𝒢    
                if in_subgoal && start_goal != end_goal && start_goal != 1 && is_connected(alg, start_goal, end_goal)
                    num_updates[start_goal, end_goal] += 1

                    # new_mean = old_mean + (datapoint - old_mean)/num_datapoints_used
                    # increment \bar{r}(g,g')
                    alg.g2g_reward_model[start_goal, end_goal] += (1/num_updates[start_goal, end_goal])*(alg.s2g_reward_models[end_goal](obs, action) - alg.g2g_reward_model[start_goal, end_goal])
                    # increment \bar{Γ}(g,g') # TODO probs in this matrix are sometimes negative
                    alg.g2g_discount_model[start_goal, end_goal] += (1/num_updates[start_goal, end_goal])*(alg.s2g_discount_models[end_goal](obs, action) - alg.g2g_discount_model[start_goal, end_goal])
                end
            end
        end
    end
    println("r̃(g,g') = ")
    println(alg.g2g_reward_model)
    println("Γ̃(g,g') = ")
    println(alg.g2g_discount_model)
end

# function learn_g2g_models(alg::gsp)
#     println("learning g2g")
#     state_space_size = size(alg.meta_data["subgoal_locations"], 1)
#     sample_size = 30

#     for starting_subgoal_idx=2:state_space_size
#         for target_subgoal_idx=1:state_space_size
#             if starting_subgoal_idx != target_subgoal_idx
#                 r_model, Γ_model = g2g(alg.meta_data["subgoal_locations"][target_subgoal_idx], alg.meta_data["initiation_radius"], alg.meta_data["subgoal_locations"][starting_subgoal_idx], alg.s2g_reward_models[target_subgoal_idx], alg.s2g_discount_models[target_subgoal_idx], sample_size, alg.option_policies[target_subgoal_idx])
#                 alg.g2g_reward_model[starting_subgoal_idx, target_subgoal_idx] = r_model
#                 alg.g2g_discount_model[starting_subgoal_idx, target_subgoal_idx] = Γ_model
#             end
#         end
#     end

# end

# function g2g(target_loc, init_radius, start_loc, reward_model, discount_model, sample_size, option_policy)

#     # check if start is not in init set of target
#     if (target_loc[1] - start_loc[1])^2 + (target_loc[2] - start_loc[2])^2 > init_radius^2
#         return 0, 0
#     end

#     # generate observations in start loc
#     reward_final = 0
#     discount_final = 0
#     for i=1:sample_size
#         r = 0.04 * sqrt(rand())
#         theta = rand() * 2 * π
#         x = start_loc[1] + r * cos(theta)
#         y = start_loc[2] + r * sin(theta)

#         δx = 0.0
#         δy = 0.0

#         observation = [x,  y, δx, δy]
#         chosen_action = option_policy(observation)

#         reward_final += reward_model(observation, chosen_action)[1]
#         discount_final += discount_model(observation, chosen_action)[1]
#     end

#     reward_final /= sample_size
#     discount_final /= sample_size

#     return reward_final, discount_final
# end

"""
    check_subgoals(observation, subgoal_locations)
Checks if a given observation is in any subgoal's green region.
"""
function check_subgoals(observation, subgoal_locations, subgoal_radius)
    for (number, loc) in enumerate(subgoal_locations)
        if sqrt((observation[1] - loc[1])^2 + (observation[2]-loc[2])^2) < subgoal_radius
            return true, number
        end
    end
    return false, nothing
end

"""
    is_connected(startgoal_number, endgoal_number, subgoal_locations, initiation_radius)
Checks if a two subgoals will be connected in the abstract MDP.
"""
function is_connected(alg, startgoal_number, endgoal_number)
    subgoal_locations = alg.meta_data["subgoal_locations"]
    initiation_radius = alg.meta_data["initiation_radius"]
    subgoal_radius = alg.meta_data["subgoal_radius"]
    if sqrt(sum((subgoal_locations[startgoal_number].-subgoal_locations[endgoal_number]).^2)) < initiation_radius # max separation between subgoals
        return true
    else
        return false
    end
end


# planning step
function plan_in_AMDP(alg::gsp)

    # reset v_AMDP
    alg.v_AMDP = zeros(size(alg.v_AMDP))


    println("planning in AMDP")
    r_table = alg.g2g_reward_model
    Γ_table = alg.g2g_discount_model
    θ = 0.0001

    # alg.v_AMDP = zeros(size(r_table, 1))

    N = size(r_table, 1)

    while true
        Δ = 0.0

        for s in 2:N
            v = alg.v_AMDP[s]

            # V(s) = max_a ∑ p(s′,r|s,a)[r + γV(s′)]
            values = zeros(N)
            for s′ in 1:N
                    values[s′] = r_table[s, s′] + Γ_table[s, s′] * alg.v_AMDP[s′]
            end
            alg.v_AMDP[s] = values[argmax(values)] 
            
    
            Δ = max(Δ, abs(v - alg.v_AMDP[s]))
        end

        if Δ < θ
            break
        end
    end
    println("V_AMDP =", alg.v_AMDP)
end



# projection step
function make_projection(alg::gsp)

    function projection(obs, action, subgoal_locations, initiation_radius, s2g_reward_models, s2g_discount_models, v_AMDP)
        max_bootstrap = -Inf
        for (idx,subgoal) in enumerate(subgoal_locations)
            if sqrt((obs[1] - subgoal[1])^2 + (obs[2]-subgoal[2])^2) < initiation_radius
                    max_bootstrap = max(max_bootstrap, s2g_reward_models[idx](obs, action) + s2g_discount_models[idx](obs, action)*v_AMDP[idx])
            end

        end
        return max_bootstrap, max_bootstrap==-Inf
    end

    return (obs, action) -> projection(obs, action, alg.meta_data["subgoal_locations"], alg.meta_data["initiation_radius"], alg.s2g_reward_models, alg.s2g_discount_models, alg.v_AMDP)
end

# # task learning
# function learn_with_gsp(alg::gsp)
#     beta = alg.meta_data["beta"]
#     projection = make_projection(alg)
#     gsp_learner = alg.agt(projection, beta)

#     num_episodes = alg.meta_data["num_episodes"]

#     for i=1:num_episodes
#         results = control_episode(gsp_learner, alg.env)

#         if i % 10 == 0
#             println("Ep ", i, " return: ", results[1], "len: ", results[3])
#         end

#     end

# end


# function gsp_progression(alg::gsp)
#     println("GSP progression")


#     println("Learning Option policies")
#     learn_option_policies(alg);

#     println("Learning s2g models")
#     learn_s2g_models(alg);

#     println("Learning g2g models")
#     learn_g2g_models(alg);

#     println("Planning in AMDP")
#     plan_in_AMDP(alg);

#     println("Learning with GSP")
#     gsp_results = learn_with_gsp(alg)


#     return gsp_results
# end