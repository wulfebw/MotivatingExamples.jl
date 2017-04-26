export
    Policy,
    UnivariateGaussianPolicy,
    MultivariateGaussianPolicy,
    step

abstract Policy
srand(policy::Policy, seed::Int) = srand(policy.rng, seed)

@with_kw type UnivariateGaussianPolicy <: Policy
    μ::Float64 = 0.
    σ::Float64 = 1.
    rng::MersenneTwister = MersenneTwister(1)
end
step(policy::UnivariateGaussianPolicy, state::Array{Float64}) = [
    randn(policy.rng) * policy.σ + policy.μ]

@with_kw type MultivariateGaussianPolicy <: Policy
    μ::Array{Float64} = [0.,0.]
    Σ::Array{Float64} = eye(2)
    d::Distribution = MvNormal(μ, Σ)
    rng::MersenneTwister = MersenneTwister(1)
end
step(policy::MultivariateGaussianPolicy, state::Array{Float64}) = rand(policy.d)

