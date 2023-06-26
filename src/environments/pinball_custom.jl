using BenchmarkEnvironments: PinBallConfig, PinballObstacle, collision, add_impulse!, pinball_sim_update!, add_drag!, checkbounds!, at_goal
using DecisionMakingEnvironments: SequentialProblem
using BenchmarkEnvironments:pinball_finitetime, read_config

using Plots
using LinearAlgebra


function custom_pinball(config::String, initiation_radius; maxT=10000, stochastic_start=false, randomize=false, num_episodes=10000000000, threshold=10000000000, subgoal_locs=[], fixed_start = nothing)
    X = zeros((4,2))
    X[1,:] .= [0., 1.]  # x range
	X[2,:] .= [0., 1.]  # y range
	X[3,:] .= [-2., 2.] # xdot range
    X[4,:] .= [-2., 2.] # ydot range
    S = ([0.0, maxT],
        X
    )
    A = 1:5
    obstacles, conf = read_config(Float64, config)


    if length(subgoal_locs) == 0 # no subgoals provided ⟹ randomly pick a valid target position
        # Setup target location
        target_pos = pick_target()
        # println(target_pos)
        while is_colliding(target_pos, obstacles, conf.target_radius)
            target_pos = pick_target()
            # println(target_pos)
        end
        # target_pos = verify_bounds(target_pos, obstacles, conf.target_radius)
    else
        target_pos = subgoal_locs[1] # use the first subgoal from list provided
    end

    if fixed_start === nothing
        # Setup start location
        if length(subgoal_locs) <= 1 # only 1 subgoal ⟹ we're generating trajectory dataset, 0 subgoals ⟹also randomly pick from the target
            start_pos = pick_start(target_pos, initiation_radius, conf.target_radius, conf.ball_radius)
            while is_colliding(start_pos, obstacles, conf.ball_radius)
                start_pos = pick_start(target_pos, initiation_radius, conf.target_radius, conf.ball_radius)
            end
        else # if we have more than one subgoal provided, we want to randomly spawn in one of their initiation set
            starting_subgoal = rand(subgoal_locs[2:end]) # if more than 1 subgoal provided, we are doing GSP and want to start in one of their initiation sets
            start_pos = pick_start(starting_subgoal, initiation_radius, conf.target_radius, conf.ball_radius) # start position is randomly picked within the initiation set of a subgoal
            while is_colliding(start_pos, obstacles, conf.ball_radius)
                start_pos = pick_start(starting_subgoal, initiation_radius, conf.target_radius, conf.ball_radius)
            end
        end
    else
        start_pos = fixed_start
    end
    # start_pos = verify_bounds(start_pos, obstacles, conf.ball_radius)


    conf = PinBallConfig(Float64, start_pos, target_pos, conf.target_radius, conf.ball_radius, conf.noise, conf.drag, conf.force)
    
    x = zeros(4)
    x .= conf.start_pos[1], conf.start_pos[2], 0.0, 0.0
    
    dt = 0.05

    p = (s,a)->pinball_step!(s, a, conf, obstacles, dt, stochastic_start, maxT, initiation_radius)
    d0 = ()->pinball_d0!(conf.start_pos, stochastic_start, conf.target_pos, initiation_radius, conf.target_radius, obstacles, conf.ball_radius, subgoal_locs, fixed_start)


    meta = Dict{Symbol,Any}()
    meta[:minreward] = -5.0
    meta[:maxreward] = 10000.0
    meta[:minreturn] = -5 * ceil(maxT / (dt * 20))  # time moves at 20*dt per step
    meta[:maxreturn] = 10000  # actually lower than this, but if you started in the goal state this would be the case. 
    meta[:stochastic] = true
    meta[:minhorizon] = 40  # not sure the the true minimum is. This seems like a good lower bound
    meta[:maxhorizon] = ceil(maxT / (dt * 20))
    meta[:discounted] = false
    meta[:episodes] = num_episodes
    meta[:threshold] = threshold
    render = (state,clearplot=false)->pinballplot2(state, obstacles, conf, subgoal_locs, initiation_radius)
    m = SequentialProblem(S,X,A,p,d0,meta,render)
    
	return m, conf.target_pos

end


function is_colliding(loc, obstacles, radius)
    for obs in obstacles
        if in_bounding_box(loc, obs, radius)
            if center_in_obs(loc, obs)
                return true
            elseif close_to_edge(loc, obs, radius)
                return true
            end
        end
    end

    return false
end

function in_bounding_box(loc, obs, radius)
    if loc[1] > obs.minx - radius && loc[1] < obs.maxx + radius && loc[2] > obs.miny - radius && loc[2] < obs.maxy + radius
        # println("In the boundding box!")
        return true
    end

    return false
end


function close_to_edge(loc, obs::PinballObstacle, radius)
    loc = collect(loc)
    N = size(obs.points, 1)

    p1 = collect(obs.points[1])

    for i in 2:(N+1)
        p2 = collect(obs.points[(i-1)%N + 1])


        α = (loc[1] - p1[1]) * (p2[1] - p1[1]) + (loc[2] - p1[2]) * (p2[2] - p1[2])
        α /= norm(p2 - p1)^2

        if α <= 0
            dist = norm(p1 - loc)
        elseif α >= 1
            dist = norm(p2 - loc)
        else
            dist = norm(p1 * (1 - α) + p2 * α - loc)
        end

        if dist < radius
            # println("Close to edge!")
            return true
        end

        p1 = p2
    end

    return false
end


function center_in_obs(loc, obs::PinballObstacle)
    N = size(obs.points, 1)
    counter = 0

    p1 = obs.points[1]

    for i in 2:(N+1)
        p2 = obs.points[(i-1)%N + 1]
        if loc[2] > min(p1[2], p2[2])
            if loc[2] <= max(p1[2], p2[2])
                if loc[1] <= max(p1[1], p2[1])
                    if p1[2] != p2[2]
                        xintercept = (loc[2] - p1[2])*(p2[1] - p1[1])/(p2[2] - p1[2]) + p1[1]
                        if p1[1] == p2[1] || loc[1] <= xintercept
                            counter += 1
                        end
                    end
                end
            end
        end
        p1 = p2
    end

    if counter % 2 == 0
        return false
    else
        # println("Center in obs!")
        return true
    end

end

function pick_start(target_pos, initiation_radius, target_radius, ball_radius)

    θ = rand() * 2π
    r = rand() * (initiation_radius - target_radius - 2 * ball_radius) + target_radius + ball_radius

    x = r * cos(θ) + target_pos[1]
    y = r * sin(θ) + target_pos[2]

    if x < 0 || x > 1 || y < 0 || y > 1
        (x, y) = pick_start(target_pos, initiation_radius, target_radius, ball_radius)
    end

    return (x, y)
end

function pick_target()
    x = rand()
    y = rand()

    return x, y
end




function pinball_step!(state, action, config, obstacles, dt, stochastic_start, maxT, initiation_radius)
    # pinball_update!($state, $action, $config, $obstacles, $dt, $maxT)
    t,x = state
    t, x, reward, γ = pinball_update!(t,x, action, config, obstacles, dt, maxT)
    # if γ == 0.0
    #     reset_ball!(x, config.start_pos, stochastic_start, config.target_pos, initiation_radius, config.target_radius, obstacles, config.ball_radius)
    #     t = 0.0
    # end

    return (t,x), x, reward, γ
    # return nothing
end

function pinball_d0!(start_pos, stochastic_start, target_pos, initiation_radius, target_radius, obstacles, ball_radius, subgoal_locs, fixed_start)
    x = zeros(4)
    reset_ball!(x, start_pos, stochastic_start, target_pos, initiation_radius, target_radius, obstacles, ball_radius, subgoal_locs, fixed_start)
    # reset_ball!(ball, start_pos, stochastic_start)
    # update_observation!(x, ball)
    t = 0.0
    return (t,x), x
end

function reset_ball!(x, start_pos::Tuple, stochastic_start::Bool, target_pos, initiation_radius, target_radius, obstacles, ball_radius, subgoal_locs, fixed_start)


    if fixed_start === nothing
        if length(subgoal_locs) <= 1 # only 1 subgoal ⟹ we're generating trajectory dataset, 0 subgoals ⟹also randomly pick from the target
            start_pos = pick_start(target_pos, initiation_radius, target_radius, ball_radius)
            while is_colliding(start_pos, obstacles, ball_radius)
                start_pos = pick_start(target_pos, initiation_radius, target_radius, ball_radius)
            end
        else # if we have more than one subgoal provided, we want to randomly spawn in one of their initiation set
            starting_subgoal = rand(subgoal_locs[2:end]) # if more than 1 subgoal provided, we are doing GSP and want to start in one of their initiation sets
            start_pos = pick_start(starting_subgoal, initiation_radius, target_radius, ball_radius) # start position is randomly picked within the initiation set of a subgoal
            while is_colliding(start_pos, obstacles, ball_radius)
                start_pos = pick_start(starting_subgoal, initiation_radius, target_radius, ball_radius)
            end
        end
    else
        start_pos = fixed_start
    end

    x[1] = start_pos[1]
    x[2] = start_pos[2]

    # x[1] = start_pos[1]
    # x[2] = start_pos[2]
    # ball.x = start_pos[1]
	# ball.y = start_pos[2]
	if stochastic_start
		# ball.x += 0.02 * randn()
        # ball.y += 0.02 * randn()
        x[1] += 0.02 * randn()
		x[2] += 0.02 * randn()
	end
	# ball.xDot = 0.
    # ball.yDot = 0.
    x[3] = 0.0
    x[4] = 0.0
    return nothing
end

function pinball_update!(t, ball, action::Int, config::PinBallConfig{T}, obstacles::Array{PinballObstacle{T},1}, dt::T, maxT) where {T}
    if action <= 0 || action > 5
        error("Action needs to be an integer in [1, 5]")
    end

	if rand() < config.noise
		action = rand(1:5)
	end
    # add action effect
    if action == 1
        add_impulse!(ball, config.force, 0.)  # Acc x
    elseif action ==2
        add_impulse!(ball, 0., -config.force) # Dec y
    elseif action ==3
        add_impulse!(ball, -config.force, 0.) # Dec x
    elseif action == 4
        add_impulse!(ball, 0., config.force) # Acc y
    else
        add_impulse!(ball, 0., 0.)  # No action
    end

    reward = 0.0
    for i in 1:20
        pinball_sim_update!(ball, dt, config.ball_radius, obstacles, i)
		t += dt
        found_goal = at_goal(ball, config)
		done = found_goal || t > maxT


        if done
            reward = (10000. * found_goal) - (1 - found_goal)

            return t,ball,reward,0.0
        end
    end

    add_drag!(ball, config.drag)
    checkbounds!(ball)

    reward = -1.0

    return t,ball,reward,1.0
end


function circle_shape(x, y, r)
	θ = LinRange(0, 2*π, 500)
	x .+ r*sin.(θ), y .+ r*cos.(θ)
end




@userplot PinBallPlot2
@recipe function f(ap::PinBallPlot2)
    state, obstacles, config, subgoal_locs, initiation_radius = ap.args

    t, ball = state
    ballx = ball[1]
    bally = ball[2]
    bradius = config.ball_radius
	tx, ty = config.target_pos
	tr = config.target_radius


	legend := false
	xlims := (0., 1.)
	ylims := (0., 1.)
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	# obstacles
	for ob in obstacles
		@series begin
			seriestype := :shape
			seriescolor := :blue
			xpts = [p[1] for p in ob.points]
			ypts = [p[2] for p in ob.points]

			xpts, ypts
		end
	end

	# goal
	@series begin
		seriestype := :shape
		linecolor := nothing
		seriescolor := :green
		aspect_ratio := 1.
		fillalpha := 0.4

		circle_shape(tx, ty, tr)
	end

    # initiation_radius
	@series begin
		seriestype := :path
		linecolor := :green
		seriescolor := nothing
		aspect_ratio := 1.

		circle_shape(tx, ty, initiation_radius)
	end

	# ball
	@series begin
		seriestype := :shape
		linecolor := nothing
		seriescolor := :red
		aspect_ratio := 1.

		circle_shape(ballx, bally, bradius)
	end


    # subgoals
    for (x, y) in subgoal_locs[2:end]

    
        @series begin
            seriestype := :shape
            linecolor := nothing
            seriescolor := :blue
            aspect_ratio := 1.
            fillalpha := 0.4

            circle_shape(x, y, tr)
        end

        # subgoal initiation_radius
        @series begin
            seriestype := :path
            linecolor := :blue
            seriescolor := nothing
            aspect_ratio := 1.

            circle_shape(x, y, initiation_radius)
        end
    end

end



function main()

    for _ in 1:100
        env, target = custom_pinball("pinball_hard.cfg", 0.5)
        s, x = env.d0()
        p = env.render(s)
        display(p)
        # sleep(1)
    end

end