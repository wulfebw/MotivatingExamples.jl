# using Base.Test
# using MotivatingExamples

function test_continuous_1D_random_walk_env()
    env = Continuous1DRandomWalkEnv()
    x = reset!(env)
    a = [0.]
    nx, r, done = step(env, a)
    @test x == nx
    @test r == 0.
    @test done == false

    srand(env, 1)
    a = reset!(env)
    srand(env, 1)
    b = reset!(env)
    @test all(a .== b)
end

function test_continuous_2D_rare_event_env()
end

println("test_environment.jl")
@time test_continuous_1D_random_walk_env()