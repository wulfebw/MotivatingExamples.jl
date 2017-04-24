export
    MCLearner,
    get_feedback,
    predict,
    compute_single_episode_state_returns,
    learn

type MCLearner <: Learner
    grid::RectangleGrid # for intepolating continuous states
    values::Array{Float64} # for maintaining state values (target dim, num unique states)
    targets::Array{Float64} # temp container for returning target values
    target_dim::Int # dimension of output
    lr::Float64 # learning rate for td update
    discount::Float64 # discount rate 
    pred_errors::Array{Float64} # prediction errors
    function MCLearner(grid::RectangleGrid, target_dim::Int;
            lr::Float64 = .1, discount::Float64 = 1.)
        values = zeros(Float64, target_dim, length(grid))
        targets = zeros(Float64, target_dim)
        pred_errors = Float64[]
        return new(grid, values, targets, target_dim, lr, discount, pred_errors)
    end
end

function get_feedback(learner::MCLearner)
    pred_errors = learner.pred_errors[:]
    empty!(learner.pred_errors)
    return pred_errors
end

# computes returns assuming the experience contains a single complete 
# or partial episode - i.e., it doesn't allow for repeated simulation of 
# a single state. It would be possible to account for multiple episodes 
# but not in a nice way, and not in a manner that would combine estimates 
# for a single state
function compute_single_episode_state_returns(experience::ExperienceMemory, 
        learner::Learner, discount::Float64)
    state_returns = Tuple{Array{Float64},Array{Float64}}[]
    if length(experience) == 0
        return state_returns
    end

    # optionally bootstrap
    x, a, r, nx, done = get(experience, length(experience))
    if done
        R = zeros(size(r))
    else
        R = predict(learner, nx)
    end

    # accumulate returns backward
    for i in (length(experience)):-1:1
        x, a, r, nx, done = get(experience, i)
        R = r + discount * R
        push!(state_returns, (x, R))
    end
    return state_returns
end

function learn(learner::MCLearner, x::Array{Float64}, ret::Array{Float64})
    # update 
    total_pred_error = 0
    inds, ws = interpolants(learner.grid, x)
    for (ind, w) in zip(inds, ws)
        # something is happening to cause this index to be 0 
        # I am not sure what it is, but do not believe it is due to 
        # GridInterpolations, so is likely a bug in this code
        if ind == 0
            continue
        end

        # update
        pred_error = w * (ret - learner.values[:, ind])
        learner.values[:, ind] += learner.lr * pred_error
        total_pred_error += pred_error
    end

    # store td-error associated with this state for later use as feedback
    push!(learner.pred_errors, sum(abs(total_pred_error)))
end

function learn(learner::MCLearner, experience::ExperienceMemory)
    state_returns = compute_single_episode_state_returns(experience, learner, 
        learner.discount)
    for (x, ret) in state_returns
        learn(learner, x, ret)
    end
    return get_feedback(learner)
end