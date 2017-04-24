export
    Uniform,
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

rand(rng::MersenneTwister, d::Uniform) = rand(rng) * (d.hi - d.lo) + d.lo
pdf(d::Uniform, v::Float64) = 1. / (d.hi - d.lo)
logpdf(d::Uniform, v::Float64) = log(pdf(d, v))

