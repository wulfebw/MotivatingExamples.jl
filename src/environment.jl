export
    Environment,
    Env,
    Continuous1DRandomWalkEnv,
    Continuous2DRareEventEnv,
    reset!,
    step,
    get_thresh

abstract Environment
typealias Env Environment
srand(env::Env, seed::Int) = srand(env.rng, seed)
# when the distribution passed in in nothing, call the default
reset!(env::Env, null_dist::Void) = reset!(env)

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
    while x[1] < env.xmin || x[1] > env.xmax
        x = rand(env.rng, dist)
    end
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
    # max thresh gives the upper bound on the action values that trigger rare event
    # selected to place .9995 of the mass of unit gaussian below
    max_thresh::Float64 = 3.290
    # likewise lower bound places .9955 of mass of unit gaussian below
    min_thresh::Float64 = 1.959
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
function get_thresh(env::Continuous2DRareEventEnv, x::Array{Float64})
    αx = abs(x[1] / env.xmax)
    αy = abs(x[2] / env.ymax)
    α = (αx + αy) / 2.
    return (env.max_thresh - env.min_thresh) * α + env.min_thresh
end
function step(env::Continuous2DRareEventEnv, a::Array{Float64})
    env.x += a
    env.x[1] = min(max(env.x[1], env.xmin), env.xmax)
    env.x[2] = min(max(env.x[2], env.xmin), env.xmax)
    r, done = [0.], false
    thresh = get_thresh(env, env.x)
    if a[1] > thresh && a[2] > thresh
        r, done = [1.], true
    end
    return (env.x, r, done)
end

function pdf(env::Continuous2DRareEventEnv, x::Vector{Float64})
    return (1 / (env.xmax - env.xmin)) * (1 / (env.ymax - env.ymin))
end
function pdf(env::Continuous2DRareEventEnv, x::Array{Float64})
    _, N = size(x)
    return ones(N) ./ ((env.xmax - env.xmin) * (env.ymax - env.ymin))
end