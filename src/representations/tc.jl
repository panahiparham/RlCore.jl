include("tile_coder.jl")

mutable struct TC
    tiles
    tilings
    _tc::TileCoder

    function TC(tiles, tilings, observations)
        dims = size(observations, 1)
        _tc = TileCoder(tilings, tiles, dims)

        new(tiles, tilings, _tc)
    end
end


function features(rep::TC)
    return feature_size(rep._tc)
end

function  encode(rep::TC, observation)
    return create_features(rep._tc, observation)
end