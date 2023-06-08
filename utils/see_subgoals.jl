using DecisionMakingEnvironments
using BenchmarkEnvironments
using DecisionMakingUtils
using Flux: Chain
using Plots
using LinearAlgebra
using JLD2
# ICLR_subgoals = [(0.5, 0.07), # Terminal state
#                 (0.8, 0.07),
#                 (0.55, 0.26),
#                 (0.67, 0.26),
#                 (0.3, 0.26),
#                 (0.65, 0.46),
#                 (0.18, 0.55),
#                 (0.38, 0.7),
#                 (0.78, 0.7)
#                 ]

# box_subgoals = [(0.85, 0.85),
                # (0.75, 0.25),
                # (0.25, 0.75)
                # ]

exp_subgoals = [(0.3, 0.3),
                (0.55, 0.45), #1
                (0.33, 0.64),#2
                (0.73, 0.80)#4
                ] # first is always the main goal


include("../environments/pinball_custom.jl")
initiation_radius = 0.5
pinball_config = "pinball_easy.cfg"
env, t = custom_pinball(pinball_config, initiation_radius; subgoal_locs=exp_subgoals, fixed_start=(0.9, 0.9))
s, x = env.d0()
p = env.render(s)
display(p)
