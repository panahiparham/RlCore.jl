include("main.jl")


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "-e", "--exp"
            help = "Experiment file"
            arg_type = String
            required = true

        "-i", "--idxs"
            help = "Indices of hyperparameters to run"
            arg_type = Int
            required = true
            nargs = '+'
    end

    return parse_args(s)
end

function as_script()
    args = parse_commandline()
    indices = args["idxs"]
    exp_file = args["exp"]

    main(exp_file, indices)

end

as_script()