export
    Environment,
    Env,
    Continuous1DRandomWalkEnv,
    reset!,
    step

abstract Environment
typealias Env Environment

@with_kw type Continuous1DRandomWalkEnv <: Environment
    xmin::Float64 = -10.
    xmax::Float64 = 10
    initial_state_dist::Distribution = Uniform(xmin, xmax)
    x::Array{Float64} = [0.]
end
function reset!(env::Continuous1DRandomWalkEnv, dist::Distribution = env.initial_state_dist) 
    x = rand(dist)
    if typeof(x) == Float64
        x = [x]
    end
    env.x = x
    return env.x
end
function reset!(env::Continuous1DRandomWalkEnv, x::Array{Float64})
    env.x = x
end
function step(env::Continuous1DRandomWalkEnv, a::Array{Float64})
    env.x += a[1]
    if env.x[1] > env.xmax
        r = 1.
        done = true
    elseif env.x[1] < env.xmin
        r = 0.
        done = true
    else
        r = 0.
        done = false
    end
    return (env.x, r, done)
end