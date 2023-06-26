using RLGlue
using Statistics
using ArgParse
using Random


include("problems/registry.jl")
include("agents/registry.jl")

include("experiment/ExperimentModel.jl")

include("utils/collector.jl")
include("utils/results.jl")

# exp -> exp_name, total_steps, episode_cutoff


function main(exp_file, indices)

    # ----------------------
    # -- Experiment Def'n --
    # ----------------------

    exp = load_experiment(exp_file) # TODO


    for idx in indices
        # TODO: setup checkpoint


    
        # setup collector
        collector = Collector(
            config = Dict(
                "step_return" => Window(100),
                # "weight_norm" => Window(100),
            ),
            idx = idx
        )
        run = get_run(exp, idx) # TODO buggggg run is 1 larger than it should be
        perm = get_permutation(exp, idx)


        # setup experiment
        problem = get_problem(exp.problem) # this needs to be modified with idx for hyper selection, why?
        env = problem.environment
        agent = get_agent(exp.agent, env.observations, env.actions, perm, run, problem.discount_factor)

        glue = Glue(env, agent)

        if glue.total_steps == 0
            start!(glue)
        end

        for step in glue.total_steps + 1:exp.total_steps
            interaction = step!(glue)

            # collect per step data

            println("step: ", glue.total_steps, " in tis ep: ", glue.num_steps, " cutoff: " , exp.episode_cutoff, " state: ", interaction.o, " action: ",interaction.a , " reward: ", interaction.r, " term: ", interaction.t)

            if interaction.t || (exp.episode_cutoff > -1 && glue.num_steps >= exp.episode_cutoff)
                # collect episodic data: step_return, steps, episodic_return
                repeat!(collector, "step_return", glue.total_reward, glue.num_steps)
                # repeat!(collector, "weight_norm", norm(glue.agent.w), glue.num_steps)
                # println("steps: ", glue.total_steps, " reward: ", glue.total_reward)

                
                start!(glue)

            end

        end

        # TODO: try to detect if a run never finishd



        # force the data to always have same length
        fill_rest!(collector, "step_return", Int(glue.total_steps / 100))
        # fill_rest!(collector, "weight_norm", Int(glue.total_steps / 100))
        reset!(collector)


        # ------------
        # -- Saving --
        # ------------
        save_collector(collector, exp)
        

    end
end

# callbacks for td 

# function give_me_td(glue)
#     td_err = r + glue.agent.γ * dot(glue.agent._weights, glue._features) - dot(glue.agent._weights, glue.agent._prev_features)


#     return td_err

# end
# function give_me_gsp_td(glue)
#     td_err = r + glue.agent.γ * dot(glue.agent._weights, glue._features) - dot(glue.agent._weights, glue.agent._prev_features)


#     return td_err

# end