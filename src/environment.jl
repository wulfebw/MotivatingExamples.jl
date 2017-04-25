export
    Environment,
    Env,
    Continuous1DRandomWalkEnv,
    Continuous2DRareEventEnv,
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


@with_kw type Continuous2DRareEventEnv <: Environment
    xmin::Float64 = -10.
    xmax::Float64 = 10.
    ymin::Float64 = -10.
    ymax::Float64 = 10.
    eps::Float64 = 1e-2 # if actions are within this value reward differs
    initial_state_dist::Distribution = MultivariateUniform(xmin, xmax, ymin, ymax)
    x::Array{Float64} = [0., 0.]
    rng::MersenneTwister = MersenneTwister(1)
end
function reset!(env::Continuous2DRareEventEnv, 
        dist::Distribution = env.initial_state_dist) 
    x = rand(env.rng, dist)
    x[1] = max(min(x[1], env.xmax), env.xmin)
    x[2] = max(min(x[2], env.xmax), env.xmin)
    env.x = x
    return env.x
end
function reset!(env::Continuous2DRareEventEnv, x::Array{Float64})
    env.x = x
end
function step(env::Continuous2DRareEventEnv, a::Array{Float64})
    env.x += a
    if (env.xmin > env.x[1] 
        || env.x[1] > env.xmax 
        || env.ymin > env.x[1] 
        || env.x[2] > env.ymax)
        done = true
    end
    if abs(a[1] - a[2]) < env.eps
        r = 1
    else
        r = 0
    end

    return (env.x, r, done)
end

function pdf(env::Continuous2DRareEventEnv, x::Vector{Float64})
    return 1 / (env.xmax - env.xmin)
end
function pdf(env::Continuous2DRareEventEnv, x::Array{Float64})
    _, N = size(x)
    return ones(N) ./ (env.xmax - env.xmin)
end