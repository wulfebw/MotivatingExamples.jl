export
    Uniform,
    MultivariateUniform,
    pdf,
    logpdf,
    rand

#=
This file reproduces some types and functions from the Distributions.jl
library. The reason for this is that Distributions.jl does not have an 
interface for passing a RNG to the rand call. For more information 
see https://github.com/JuliaStats/Distributions.jl/issues/197
Passing an rng the rand calls seems to be the best way to get 
deterministic results. For example, the Base module of julia does it this way. 
So I've just reimplemented a few types with this functionality and have not 
imported Distributions.jl
=#

type Uniform <: Distribution
  hi::Float64
  lo::Float64  
end

rand(d::Uniform) = rand() * (d.hi - d.lo) + d.lo
rand(d::Uniform, n_samples::Int) = rand(n_samples) .* (d.hi - d.lo) .+ d.lo
rand(rng::MersenneTwister, d::Uniform) = rand(rng) * (d.hi - d.lo) + d.lo
pdf(d::Uniform, v::Float64) = 1. / (d.hi - d.lo)
logpdf(d::Uniform, v::Float64) = log(pdf(d, v))
pdf(d::Uniform, v::Array{Float64}) = ones(length(v)) ./ (d.hi - d.lo)
logpdf(d::Uniform, v::Array{Float64}) = log(pdf(d, v))

type MultivariateUniform <: Distribution
    x::Uniform
    y::Uniform
end
MultivariateUniform(xlo, xhi, ylo, yhi) = MultivariateUniform(
    Uniform(xlo, xhi), Uniform(ylo, yhi))
rand(rng::MersenneTwister, d::MultivariateUniform) = [rand(rng, d.x), rand(rng, d.y)]
pdf(d::MultivariateUniform, v::Array{Float64}) = pdf(d.x, v[1]) * pdf(d.y, v[2])
logpdf(d::MultivariateUniform, v::Array{Float64}) = log(pdf(d, v))

