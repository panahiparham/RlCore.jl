using JSON
using RLGlue

include("../environments/mountainCar.jl")

struct MountainCarProblem
    environment::RLGlue.BaseEnvironment
    discount_factor::Float64

    function MountainCarProblem(discount_factor::Float64 = 1.0)
        environment = MountainCar()
        discount_factor = discount_factor
        @boundscheck (discount_factor >= 0 && discount_factor <= 1)

        return new(environment, discount_factor)
    end
end
