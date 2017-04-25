using Base.Test
using MotivatingExamples

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
    srand(0)
    env = Continuous2DRareEventEnv()
    x = reset!(env)

    x = [0.,0.]
    reset!(env, x)
    thresh = get_thresh(env, x)
    @test thresh == env.min_thresh

    x = [env.xmax,env.ymax]
    reset!(env, x)
    thresh = get_thresh(env, x)
    @test thresh == env.max_thresh
    
    reset!(env, [0.,0.])
    a = [1., 1.]
    nx, r, done = step(env, a)
    @test all(nx .== a)
    @test r == [0.]
    @test done == false

    reset!(env, [0.,0.])
    a = [4., 4.]
    nx, r, done = step(env, a)
    @test all(nx .== a)
    @test r == [1.]
    @test done == true

end

println("test_environment.jl")
@time test_continuous_2D_rare_event_env()
@time test_continuous_1D_random_walk_env()