using Base.Test
using MotivatingExamples

function runtests()
    include("test_policy.jl")
    include("test_environment.jl")
    include("test_experience.jl")
    include("test_learner.jl")
    include("test_trainer.jl")
    println("\nAll tests pass!")
end

@time runtests()