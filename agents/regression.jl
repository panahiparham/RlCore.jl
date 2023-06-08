using Random

"""
Ordinary Least Squares Regression
"""
mutable struct OLS{TW, TB}
    W::TW
    ψ::TB
    function OLS(num_features, num_actions, ϕ)
        W = zeros(num_features * num_actions)
        ψ = (X) -> ξ(X, ϕ, num_features, num_actions)
        return new{typeof(W), typeof(ψ)}(W, ψ)
    end
end

"""
ξ(inputs, ϕ, num_features, num_actions)
    Returns stacked features from an array of input observations.
"""
function ξ(inputs, ϕ, num_features, num_actions)
    stacked_features = zeros(size(inputs, 1), num_features * num_actions)

    for (i,item) in enumerate(inputs)
        stacked_features[i, (item[2] - 1) * num_features + 1 : item[2] * num_features] = ϕ(item[1])
    end
    return stacked_features
end

"""
train!(alg::OLS, inputs, outputs)
    Performs orindary least squares regression on observation dataset.
"""
function train!(alg::OLS, inputs, outputs)
    X = alg.ψ(inputs)
    y = zeros(size(outputs, 1))

    for (i, item) in enumerate(outputs)
        y[i] = item
    end
    alg.W = X \ y
end

"""
predict(alg::OLS, inputs)
    Returns predictions for a given set of inputs.
"""
function predict(alg::OLS, inputs)
    X = alg.ψ(inputs)
    return X * alg.W
end

"""
validate(alg::OLS, inputs, outputs)
    Validates predictions by calculating R^2 score.
"""
function validate(alg::OLS, inputs, outputs)
    preds = predict(alg, inputs)
    return r2_score(outputs, preds)
end

"""
r2_score(y_true, y_pred)
    Computes coefficient of determination for our current predictions.
"""
function r2_score(y_true, y_pred)
    SS_res = sum((y_true - y_pred).^2)
    SS_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - SS_res / SS_tot
end


"""
Stochastic Gradient Descent Regression
"""
mutable struct SGD_regression{TW, TB}
    W::TW
    ϕ::TB
    function SGD_regression(num_features, num_actions, ϕ)
        W = zeros(num_features, num_actions)
        return new{typeof(W), typeof(ϕ)}(W, ϕ)
    end

end

"""
Gradient Descent Regression with Adam
"""
mutable struct Adam_regression{TW, TB}
    W::TW
    ϕ::TB
    function Adam_regression(num_features, num_actions, ϕ)
        W = zeros(num_features, num_actions)
        return new{typeof(W), typeof(ϕ)}(W, ϕ)
    end

end

"""
train(alg::SGD_regression, inputs, outputs, batch_size, learning_rate)
    trains s2g models using SGD updates
"""
function train(alg::SGD_regression, inputs, outputs, batch_size, learning_rate)
    inputs, outputs = shuffle_data(inputs, outputs)

    for i in 1:batch_size:size(inputs, 1)
        batch_inputs = inputs[i:min(i+batch_size-1, end), :]
        batch_outputs = outputs[i:min(i+batch_size-1, end), :]

        grad = zeros(size(alg.W))

        for (data, y) in zip(batch_inputs, batch_outputs)
            state = data[1]
            action = data[2]
            features = alg.ϕ(state)

            grad[:, action] += features * (dot(features, alg.W[:, action]) - y)
        end

        alg.W -= learning_rate * grad / batch_size

    end
end

"""
train(alg::SGD_regression, inputs, outputs, batch_size, learning_rate)
    trains s2g models using SGD updates with Adam
"""
function train(alg::Adam_regression, inputs, outputs, batch_size, learning_rate)
    inputs, outputs = shuffle_data(inputs, outputs)

    β1 = 0.9
    β2 = 0.999
    ϵ = 1e-8
    
    m = zeros(size(alg.W))
    v = zeros(size(alg.W))
    t = 0

    # preallocate m hat and v hat

    for i in 1:batch_size:size(inputs, 1)
        batch_inputs = inputs[i:min(i+batch_size-1, end), :]
        batch_outputs = outputs[i:min(i+batch_size-1, end), :]

        grad = zeros(size(alg.W))

        for (data, y) in zip(batch_inputs, batch_outputs)


            state = data[1]
            action = data[2]
            features = alg.ϕ(state)

            grad[:, action] += features * (dot(features, alg.W[:, action]) - y)
        end

        t += 1
        grad = grad / batch_size
        m = β1 * m + (1 - β1) * grad
        v = β2 * v + (1 - β2) * grad.^2

        m̂ = m / (1 - β1^t)
        v̂ = v / (1 - β2^t)

        alg.W -= learning_rate * m̂ ./ (sqrt.(v̂) .+ ϵ) # use @. for broadcasting (less allocations)

    end
end

"""
test(alg, inputs, outputs)
    algorithm agnostic testing function
"""
function test(alg, inputs, outputs)
    loss = 0.0
    for (data, y) in zip(inputs, outputs)
        state = data[1]
        action = data[2]
        features = alg.ϕ(state)
        loss += (dot(features, alg.W[:, action]) - y)^2
    end
    return loss / size(inputs, 1)
end

"""
r2score(alg, inputs, outputs)
    algorithm agnostic R^2 score computation
"""
function r2score(alg, inputs, outputs)
    ŷ = zeros(size(outputs))
    for (i, data) in enumerate(inputs)
        state = data[1]
        action = data[2]
        features = alg.ϕ(state)
        ŷ[i] = dot(features, alg.W[:, action])
    end

    res = sum(@. (outputs - ŷ) ^ 2)
    ave = mean(outputs)
    tot = sum(@. (outputs - ave) ^ 2)
    r2 = 1 - (res / tot)

    return r2
    
end

"""
shuffle_data(inputs, outputs)
    shuffles dataset
"""
function shuffle_data(inputs, outputs)
    idx = randperm(size(inputs, 1))
    return inputs[idx, :], outputs[idx, :]
end

"""
predict(alg::SGD_regression, inputs)
    returns prediciton using the model learnt via SGD
"""
function predict(alg::SGD_regression, inputs)
    features = [alg.ϕ(input[1]) for input in inputs]
    return [dot(feature, alg.W[:, inputs[i][2]]) for (i, feature) in enumerate(features)]
end

"""
predict(alg::Adam_regression, inputs)
    returns prediciton using the model learnt via SGD with Adam
"""
function predict(alg::Adam_regression, inputs)
    features = [alg.ϕ(input[1]) for input in inputs]
    return [dot(feature, alg.W[:, inputs[i][2]]) for (i, feature) in enumerate(features)]
end