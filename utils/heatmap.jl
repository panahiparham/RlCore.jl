using BenchmarkEnvironments: PinBallConfig, PinballObstacle, read_config
using Plots

include("../environments/pinball_custom.jl")



@userplot PinballHeatmap
@recipe function f(ap::PinBallHeatmap)
	obstacles, config, states, values, target_location = ap.args

	heatmap_x = [x[1] for x in states]
	heatmap_y = [x[2] for x in states]
	heatmap_z = values

	# tx, ty = config.target_pos
	tx, ty = target_location[1], target_location[2]
	tr = config.target_radius

	legend := false
	xlims := (0., 1.)
	ylims := (0., 1.)
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	@series begin
		seriestype := :heatmap
		linecolor := nothing
		seriescolor := :viridis
		aspect_ratio := 1.
		fillalpha := 0.8

		heatmap_x, heatmap_y, heatmap_z
	end


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
		seriescolor := :red
		aspect_ratio := 1.
		fillalpha := 1.0

		circle_shape(tx, ty, tr)
	end

end



@userplot PinBallHeatmapGrid
@recipe function f(ap::PinBallHeatmapGrid)
    obstacles, config, values, target_location = ap.args

    resolution = size(values, 1)
    heatmap_x = 0:1/resolution:1
    heatmap_y = 0:1/resolution:1
    heatmap_z = values

	# tx, ty = config.target_pos
	tx, ty = target_location[1], target_location[2]
	tr = config.target_radius

	legend := false
	xlims := (0., 1.)
	ylims := (0., 1.)
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

    @series begin
        seriestype := :heatmap
        linecolor := nothing
        seriescolor := :viridis
        aspect_ratio := 1.
        fillalpha := 0.8

        heatmap_x, heatmap_y, heatmap_z
    end


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
		seriescolor := :red
		aspect_ratio := 1.
		fillalpha := 1.0

		circle_shape(tx, ty, tr)
	end

end

function state_value_heatmap(meta_data, resolution, v)
	values = zeros(resolution, resolution)
	for i in 1:resolution
		for j in 1:resolution
			s = [i/resolution, j/resolution, 0.0, 0.0]
			values[i, j] = v(s)
		end
	end
	obstacles, conf = read_config(Float64, meta_data["env_config_file"])
	render = (clearplot=false)->pinballheatmapgrid(obstacles, conf, values, meta_data["subgoal_locations"][1])
	render()
end


# Simple Sample
function get_heatmap_values(q, resolution)

    values = zeros(resolution, resolution)
    for i in 1:resolution
        for j in 1:resolution
            values[i, j] = sin(i) + cos(j)
        end
    end

    return values
end

function test()

    # values = rand(100, 100) * 5
    values = get_heatmap_values(nothing, 50)
    obstacles, conf = read_config(Float64, "pinball_easy.cfg")

    render = (clearplot=false)->pinballheatmapgrid(obstacles, conf, values, (0.3, 0.3))
    render()
end
