using Plots

_BASE_PATH = joinpath(@__DIR__, "..", "..")

include(joinpath(_BASE_PATH, "src/experiment/ExperimentModel.jl"))
include(joinpath(_BASE_PATH, "src/utils/results.jl"))


function main()

    desc_paths = [
        "MCTC.json",
        "Pinball.json"
    ]


    for exp_name in desc_paths
        println("----")
        println("Loading results from $exp_name")

        results_data = load_results(joinpath(@__DIR__, exp_name))

        # merge different runs together
        merge_runs!(results_data)

        # pick best hyperparameter setting
        bidx = pick_best_idx(results_data, "step_return")
        bconfig = get_permutation(results_data._exp, bidx)
        
        println("----")
        println("Best idx: ", bidx)
        println("Best config: ", bconfig)

        println("----")
        println("Plotting...")

        lo, line, high = get_learning_curve(results_data, "step_return", bidx);

        f = plot(
            line, ribbon=(line-lo, high-line),
            fillalpha=0.2, 
            label="$(results_data._exp.description)",
            xlabel="step x 100",
            ylabel="step return",
            );

        savefig(f, joinpath(@__DIR__, "$(results_data._exp.description).png"))


        lo, line, high = get_learning_curve(results_data, "weight_norm", bidx);

        f = plot(
            line, ribbon=(line-lo, high-line),
            fillalpha=0.2, 
            label="$(results_data._exp.description)",
            xlabel="step x 100",
            ylabel="step return",
            );

        savefig(f, joinpath(@__DIR__, "$(results_data._exp.description)_weights.png"))

    end
   
end


main()




# path_to_exp = joinpath(_EXP_PATH, exp_name)

# println("----")
# println("Loading results from $path_to_exp")

# results_data = load_results(path_to_exp)


# # merge different runs together
# merge_runs!(results_data)

# # pick best hyperparameter setting
# bidx = pick_best_idx(results_data, "step_return")
# bconfig = get_permutation(results_data._exp, bidx)

# println("----")
# println("Best idx: ", bidx)
# println("Best config: ", bconfig)

# println("----")
# println("Plotting...")

# lo, line, high = get_learning_curve(results_data, "step_return", bidx);

# f = plot(
#     line, ribbon=(line-lo, high-line),
#     fillalpha=0.2, 
#     label="Random Agent",
#     xlabel="step x 1000",
#     ylabel="step return",
    
#     );

# savefig(f, joinpath(_EXP_PATH, "$(bconfig["problem"])_$(bconfig["agent"])_step-return.png"))
