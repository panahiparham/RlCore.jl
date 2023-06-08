include("pinball_problem.jl")

"""
Preset Prolems are stored in subfolders of this directory.
Path Scheme: src/problems/base_problem/start_goal.jl (for easy/)

Problem Name Scheme: relative path from src/problems/ to file excluding .json extension

For example: src/problems/easy/small_fixed.jl is pinball_easy
with random start state in a small initiation radius around a fixed goal

"""


PROBLEMS_DIR = "src/problems/";



"""
problem_name Scheme: relative path from src/problems/ to file excluding .json extension
"""
function get_problem(problem_name::String)
    config_file_path = PROBLEMS_DIR * problem_name * ".json"
    @assert isfile(config_file_path) "$problem_name is not a valid problem name"

    return PinballProblem(config_file_path)
end





function get_problem_info(problem_name)
    problem = get_problem(problem_name);

    println(
    """problem name: $problem_name
    random_start: $(problem.random_start)
    start_location: $(problem.start_location)
    initiation_radius: $(problem.initiation_radius)

    random_goal: $(problem.random_goal)
    goal_location: $(problem.goal_location)

    discount_factor: $(problem.discount_factor)
    """);
end


function get_problem_render(problem_name)
    problem = get_problem(problem_name);
    start!(problem.environment)
    render(problem.environment)
end