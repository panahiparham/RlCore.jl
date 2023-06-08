using ArgParse
using ResumableFunctions
using Random
using ProgressBars

_BASE_PATH = "/"*joinpath(split(@__DIR__, "/")[1:end-1])

include(joinpath(_BASE_PATH,"src/experiment/ExperimentModel.jl"))
include(joinpath(_BASE_PATH,"src/utils/runner.jl"))

include(joinpath(_BASE_PATH,"src/main.jl"))


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--runs"
            arg_type = Int
            required = true
        "-e"
            arg_type = String
            required = true
            nargs = '+'     
    end

    return parse_args(s)
end


function main()

    args = parse_commandline()

    # find missing indices
    e_to_missing = gather_missing_indices(args["e"], args["runs"], "step_return")
    n = values(e_to_missing) .|> length |> sum

    println("\n Missing indices: $n")

    for path in args["e"]
        println("\n Current Experiment: $path")
        indices = e_to_missing[path]

        Threads.@threads for i in ProgressBar(indices)
            main(path, [i])
        end
    end
    println("\n Done!")

end

main()