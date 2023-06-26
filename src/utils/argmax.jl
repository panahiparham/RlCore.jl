using Random


function _argmax(x::Array{T, 1}, rng = Random.default_rng()) where T <: Real
    # get max value
    max_val = maximum(x)

    # get indices of max value
    indices = findall(x .== max_val)

    # return random index
    return indices[rand(rng, 1:length(indices))]
end