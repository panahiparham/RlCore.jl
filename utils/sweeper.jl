using ResumableFunctions

@resumable function sweeper(config)
    sweep_config = config["sweep"]
    fixed_config = config["fixed"]


    # get all keys
    ks = collect(keys(sweep_config))
    # get all values
    vs = collect(values(sweep_config))

    # get all combinations of values
    for v in Iterators.product(vs...)
        # create a dictionary from the keys and values
        @yield merge(Dict(zip(ks, v)), fixed_config)
    end
end



function num_permutations(config)
    sweep_config = config["sweep"]
    vs = collect(values(sweep_config))
    return length(collect(Iterators.product(vs...)))
end





function test()


    cfg = Dict(
    "fixed" => Dict(
        "γ" => 0.99,
        "λ" => 0.9,
    ),
    "sweep" => Dict(
        "α" => [1e-3, 1e-5, 1e-7],
    )
    )


    for (id, conf) in enumerate(sweeper(cfg))
        println(id, " ", conf)
    end
end


