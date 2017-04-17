__precompile__(true)
module MotivatingExamples

using Reexport

@reexport using Distributions
@reexport using GridInterpolations
@reexport using Parameters

import Base: step, length, reset, get, done, info

include("utils.jl")
include("budget_timer.jl")
include("environment.jl")
include("experience.jl")
include("gmm.jl")
include("learner.jl")
include("monitor.jl")
include("policy.jl")
include("trainer.jl")

end # module