export
    Learner,
    TDLearner,
    reinitialize,
    get_feedback,
    predict,
    learn,
    update_feedback,
    feedback_length,
    copy_feedback

abstract Learner

reinitialize(learner::Learner) = fill!(learner.values, 0)

function predict(learner::Learner, state::Vector{Float64})
    for tidx in 1:learner.target_dim
        inds, ws = interpolants(learner.grid, state)
        learner.targets[tidx] = dot(learner.values[tidx, inds], ws)
    end
    return learner.targets
end

function predict(learner::Learner, states::Array{Float64})
    state_dim, num_states = size(states)
    values = zeros(learner.target_dim, num_states)
    for sidx in 1:num_states
        values[:, sidx] = predict(learner, states[:, sidx])
    end
    return values
end

function copy_feedback(feedback::Dict{String, Array{Float64}})
    new_feedback = Dict{String, Array{Float64}}()
    new_feedback["errors"] = feedback["errors"][:,:]
    new_feedback["states"] = feedback["states"][:,:]
    return new_feedback
end

function feedback_length(feedback::Dict{String, Array{Float64}})
    if in("errors", keys(feedback))
        return length(feedback["errors"])
    else
        return 0
    end
end

function update_feedback(feedback::Dict{String, Array{Float64}}, error::Float64,
    x::Array{Float64})

    k = "states"
    if !in(k, keys(feedback))
        feedback[k] = reshape(x, length(x), 1)
    else
        feedback[k] = hcat(feedback[k], reshape(x, length(x), 1))
    end

    k = "errors"
    if !in(k, keys(feedback))
        feedback[k] = [error]
    else
        feedback[k] = hcat(feedback[k], error)
    end
    return feedback
end

function update_feedback(dest::Dict{String, Array{Float64}},
        src::Dict{String, Array{Float64}})
    for k in ["states", "errors"]
        if !in(k, keys(dest))
            dest[k] = src[k]
        else
            dest[k] = hcat(dest[k], src[k])
        end
    end
    return dest
end

function get_feedback(learner::Learner)
    feedback = copy_feedback(learner.feedback)
    empty!(learner.feedback)
    return feedback
end

type TDLearner <: Learner
    grid::RectangleGrid # for intepolating continuous states
    values::Array{Float64} # for maintaining state values (target dim, num unique states)
    targets::Array{Float64} # temp container for returning target values
    target_dim::Int # dimension of output
    lr::Float64 # learning rate for td update
    discount::Float64 # discount rate 
    feedback::Dict{String, Array{Float64}} # reporting errors
    function TDLearner(grid::RectangleGrid, target_dim::Int;
            lr::Float64 = .1, discount::Float64 = 1.)
        values = zeros(Float64, target_dim, length(grid))
        targets = zeros(Float64, target_dim)
        feedback = Dict{String, Array{Float64}}()
        return new(grid, values, targets, target_dim, lr, discount, feedback)
    end
end

function learn(learner::TDLearner, x::Array{Float64}, r::Array{Float64}, 
        nx::Array{Float64}, done::Bool)
    # update 
    total_td_error = 0
    inds, ws = interpolants(learner.grid, x)
    for (ind, w) in zip(inds, ws)
        # something is happening to cause this index to be 0 
        # I am not sure what it is, but do not believe it is due to 
        # GridInterpolations, so is likely a bug in this code
        if ind == 0
            continue
        end

        # target value
        target = r
        if !done
            target += learner.discount * predict(learner, nx)
        end

        # update
        td_error = w * (target - predict(learner, x))
        learner.values[:, ind] += learner.lr * td_error
        total_td_error += td_error
    end

    # store error and state for later feedback
    update_feedback(learner.feedback, sum(abs(total_td_error)), x)
end

function learn(learner::TDLearner, experience::ExperienceMemory)
    for i in 1:length(experience)
        x, a, r, nx, done = get(experience, i)
        learn(learner, x, r, nx, done)
    end
    return get_feedback(learner)
end