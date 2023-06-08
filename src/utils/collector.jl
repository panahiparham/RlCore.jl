using Statistics
using DataStructures



# --------------
# -- Samplers --
# --------------

abstract type Sampler end


mutable struct Window <: Sampler
    _block::Vector{Float64}
    _clock::Int
    _size::Int

    function Window(size::Int)
        _block = zeros(Float64, size)
        _clock = 0
        _size = size

        new(_block, _clock, _size)
    end
end

function next!(window::Window, value::Float64)
    window._block[window._clock + 1] = value
    window._clock += 1

    if window._clock == window._size
        window._clock = 0
        m = mean(window._block)
        return m
    end
end

function repeat!(window::Window, value::Float64, times::Int)

    vs = Vector{Float64}()

    # repeat until block is full
    r = window._size - window._clock
    r = min(r, times)

    window._block[window._clock + 1:window._clock + r] .= value
    window._clock = (window._clock + r) % window._size
    times -= r

    if window._clock == 0
        m = mean(window._block)
        push!(vs, m)
    end

    # how many full blocks can we fill?
    d = div(times, window._size)
    for _=1:d
        push!(vs, value)
    end

    # remainder
    r = times % window._size
    if r > 0
        window._block[1:r] .= value
        window._clock = r
    end

    return vs
end

function finish!(window::Window)
    out = nothing
    if window._clock > 0
        out = mean(window._block[1:window._clock + 1])
    end

    window._clock = 0
    return out
end


mutable struct Subsample <: Sampler
    _clock::Int
    _freq::Int

    function Subsample(freq::Int)
        _clock = 0
        _freq = freq

        new(_clock, _freq)
    end
end

function next!(sub::Subsample, value::Float64)
    tick = sub._clock == 0
    sub._clock = (sub._clock + 1) % sub._freq

    if tick
        return value
    end
end

function repeat!(sub::Subsample, value::Float64, times::Int)
    vs = Vector{Float64}()
    for i=sub._clock:sub._clock + times - 1
        if i % sub._freq == 0
            push!(vs, value)
        end
    end

    sub._clock = (sub._clock + times) % sub._freq
    return vs
end

function finish!(sub::Subsample)
    sub._clock = 0
    return nothing
end


# ---------------
# -- Collector -- 
# ---------------


mutable struct Collector
    _name_idx_data::Dict{String, Dict{Int, Vector{Float64}}}
    _samplers::Dict{String, Sampler}
    _idx::Int
    function Collector(;config::Dict=Dict(), idx::Int=nothing)
        
        @assert typeof(idx) == Int
        _idx = idx
        _name_idx_data = Dict{String, Dict{Int, Vector{Float64}}}()
        _samplers = Dict{String, Sampler}()
        for (name, f) in config
            @assert isa(f, Sampler)
            _samplers[name] = f
            _name_idx_data[name] = Dict(_idx => Vector{Float64}())

        end
        new(_name_idx_data, _samplers, _idx)
    end

end

function collect!(collector::Collector, name::String, value::Any)
    v = next!(collector._samplers[name], value)
    if v === nothing
        return
    end
    idx = collector._idx
    arr = collector._name_idx_data[name][idx]
    push!(arr, v)
end

function repeat!(collector::Collector, name::String, value::Any, times::Int)
    vs = repeat!(collector._samplers[name], value, times)
    idx = collector._idx
    arr = collector._name_idx_data[name][idx]
    for v in vs
        push!(arr, v)
    end
end

function fill_rest!(collector::Collector, name::String, steps::Int)
    idx = collector._idx
    arr = collector._name_idx_data[name][idx]
    
    if length(arr) == 0
        return
    end

    l = last(arr)

    for _=length(arr):steps - 1
        push!(collector._name_idx_data[name][idx], l)
    end
end


function reset!(collector::Collector)
    for name in keys(collector._name_idx_data)

        if !(name in keys(collector._samplers))
            continue
        end

        v = finish!(collector._samplers[name])
        if v === nothing
            continue
        end
        idx = collector._idx

        arr = collector._name_idx_data[name][idx]
        # println("Resetting $name with $v") # TODO: what is this?
    end
    
end