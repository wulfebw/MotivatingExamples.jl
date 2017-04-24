export
    Environment,
    Env,
    Continuous1DRandomWalkEnv,
    reset!,
    step

abstract Environment
typealias Env Environment
srand(env::Env, seed::Int) = srand(env.rng, seed)

@with_kw type Continuous1DRandomWalkEnv <: Environment
    xmin::Float64 = -10.
    xmax::Float64 = 10
    initial_state_dist::Distribution = Uniform(xmin, xmax)
    x::Array{Float64} = [0.]
    rng::MersenneTwister = MersenneTwister(1)
end
function reset!(env::Continuous1DRandomWalkEnv, 
        dist::Distribution = env.initial_state_dist) 
    x = rand(env.rng, dist)
    if typeof(x) == Float64
        x = [x]
    end
    x = max(min(x, env.xmax), env.xmin)
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

function pdf(env::Continuous1DRandomWalkEnv, x::Vector{Float64})
    return 1 / (env.xmax - env.xmin)
end
function pdf(env::Continuous1DRandomWalkEnv, x::Array{Float64})
    _, N = size(x)
    return ones(N) ./ (env.xmax - env.xmin)
end