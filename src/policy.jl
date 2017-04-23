export
    Policy,
    UnivariateGaussianPolicy,
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

