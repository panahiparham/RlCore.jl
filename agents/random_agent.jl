

struct RandomAgent{TB, TW}
    ϕ::TB
    W::TW


    function RandomAgent(num_actions, num_features, ϕ)
        W = zeros(num_features, num_actions)
        return new{typeof(ϕ),typeof(W)}(ϕ, W)
    end

end


function random_action(alg::RandomAgent, s)
    return rand(1:size(alg.W, 2))
end

function random_beginepisode(alg::RandomAgent, s)
    return random_action(alg, s)
end

function random_update(alg::RandomAgent, s, a, r, s′, γ)
    return random_action(alg, s′)
end

