__precompile__(true)
module MotivatingExamples

using Reexport

@reexport using GridInterpolations
@reexport using Parameters
using PGFPlots

import Base: step, length, reset, get, done, info, rand, srand

include("utils.jl")
include("distributions.jl")
include("budget_timer.jl")
include("environment.jl")
include("experience.jl")
include("gmm.jl")
include("learner.jl")
include("mc_learner.jl")
include("monitor.jl")
include("policy.jl")
include("trainer.jl")
include("plotting.jl")

end # module