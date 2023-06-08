using Statistics
using Distributions
using LinearAlgebra
using ProgressBars
using Plots
using JLD2

function cumul(array)
    s = 0
    r = []
    for i in array
        s += i
        push!(r, s)
    end
    return r
end

function smooth(array, n)
    rs = []
    for (i,r) in enumerate(array)
        if i%10000 ==0
            n′ = min(n, length(array[begin:i]))
            push!(rs, sum(array[i-n′+1:i])/length(array[i-n′+1:i]))
        end
    end
    return rs
end

function ewma(data, beta)
    new_data = zeros(size(data))
    # println(size(data))

    rate = 0.0
    for step in 1:size(data)[1]
        rate = beta * rate + (1 - beta) * data[step]
        new_data[step] = rate / (1 - beta ^ step)
    end

    return new_data
end

# function ewma_runs(raw)
#     f = plot(xlabel="timestep", ylabel="reward rate", legend=false);
#     x = ewma(raw[2,:,:],0.9)
#     confidence = 0.95
#     println(size(x))
#     s = std(x, dims=1)*(quantile(TDist(size(x)[1]-1), 1 - (1-confidence/2)))
#     plot!(f, mean(x, dims=1)[begin:10000:end], ribbon=s[begin:10000:end], color="blue")
#     savefig("results/exp0/ind_runs_startinrad_e-5_movavg.png")
# end

function ewma_plotter(results_path, plot_every=1000)

    res = load_object(results_path)
    raw = res["rewards"]
    betas = res["beta"]

    num_runs = size(raw)[2]
    # plot_every = 1000
    # f = plot(ylabel="Reward Rate", yguidefontsize=32)
    f = plot(ylabel="Reward Rate", xlabel="x$plot_every timesteps")

    # ylims!(-5, 100)  # set y-axis limits to 0-1
    # plot!(legendfont=font(26), tickfont=font(26))
    # annotate!(1, 80, text("α=1e-5", 32, :left))
    # plot!(legend=false)

    for beta in 1:(size(raw)[1])
        for run in 1:size(raw)[2]
            raw[beta, run, :] = ewma(raw[beta, run, :], 0.9999)
        end

        std_err = std(raw[beta, :, :], dims=1)/sqrt(num_runs)

        # Student t-Distribution
        d = TDist(num_runs-1)
        confidence = 0.95
        sig_level = 1 - confidence
        t_value = quantile(d, 1 - (sig_level/2))
        std_err *= t_value

        # +1 because we dont want to start at the 0 reward at t = 0
        # also makes sure the x-acis terminates at 150k timesteps, and not 150k + plot_every

        # if beta==1
        #     c=1
        # elseif beta ==2
        #     c=3
        # elseif beta == 3
        #     c=4
        # end

        plot!(f, mean(raw[beta, :, :], dims=1)[begin+1:plot_every:end], ribbons=std_err[begin+1:plot_every:end], label="β=$(betas[beta])", grid=false, dpi=600)
    end
    # display(f)
    savefig("results/exp0/cedar_test.png")
end


#past this into repl
# res = load_object("results/exp0/exp_data-pinball_easy-10-runs_300000-steps_startinrad.jld2")
# raw = res["rewards"]

# f = plot(xlabel="timestep", ylabel="reward rate", legend=false)

# to plot reward "density" in time, remember to clear plot each time
# f = plot(xlabel="timestep", ylabel="reward rate", legend=false);
# for run in 1:30
#     plot!(f, cumul(raw[2, run, :]), alpha=0.1, color="blue")
# end; savefig("results/exp0/ind_runs_initrad1.png")

function exp4_plotter(results_path)
    # this is a temporary plotting function I cooked up for experiment 4
    # will tidy it and make it nicer later

    plot_every = 1000
    f = plot(xlabel="x $plot_every timestep", ylabel="reward rate")

    # plot beta = 0 from separate file
    no_gsp = load_object("results/exp0/for_exp4_exp_data-pinball_easy-30-runs_300000-steps.jld2")
    no_gsp = no_gsp["rewards"]
    num_runs = size(no_gsp)[2]
    for run in 1:num_runs
        no_gsp[1, run, :] = ewma(no_gsp[1, run, :], 0.9999)
    end
    std_err = std(no_gsp[1, :, :], dims=1)/sqrt(num_runs)
    plot!(f, mean(no_gsp[1, :, :], dims=1)[begin+1:plot_every:end], ribbons=std_err[begin+1:plot_every:end], label="β=0.0", fillalpha=0.3, grid=false)

    raw = load_object(results_path)
    τs = [2, 8, 32, 128, 512] # this needs to be changed depending on metadata in exp4

    num_runs = size(raw)[2]
    # confidence = 0.95

    for τ_size in 1:(size(raw)[1])
        for run in 1:size(raw)[2]
            raw[τ_size, run, :] = ewma(raw[τ_size, run, :], 0.9999)
        end

        # d = TDist(num_runs-1)
        std_err = std(raw[τ_size, :, :], dims=1)/sqrt(num_runs)
        # sig_level = 1 - confidence
        # t_value = quantile(d, 1 - (sig_level/2))
        # std_err *= t_value

        # +1 because we dont want to start at the 0 reward at t = 0
        # also makes sure the x-acis terminates at 150k timesteps, and not 150k + plot_every

        # if beta==1
        #     c=1
        # elseif beta ==2
        #     c=3
        # elseif beta == 3
        #     c=4
        # end

        plot!(f, mean(raw[τ_size, :, :], dims=1)[begin+1:plot_every:end], ribbons=std_err[begin+1:plot_every:end], label="size=$(τs[τ_size])", linealpha=0.5, fillalpha=0.1, grid=false, linestyle=:dash)
    end

    # plot beta = 0 from separate file
    # no_gsp = load_object("results/exp4/dataset2048.jld2")
    # num_runs = size(no_gsp)[2]
    # for run in 1:num_runs
    #     no_gsp[1, run, :] = ewma(no_gsp[1, run, :], 0.9999)
    # end
    # std_err = std(no_gsp[1, :, :], dims=1)/sqrt(num_runs)
    # plot!(f, mean(no_gsp[1, :, :], dims=1)[begin+1:plot_every:end], ribbons=std_err[begin+1:plot_every:end], label="size=2048", linealpha=0.5, fillalpha=0.1, grid=false, linestyle=:dash)

    display(f)
    # savefig("results/exp0/alpha1e-4good.png")
end
