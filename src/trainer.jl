export
    Trainer,
    AdaptiveTrainer,
    finished_training,
    prepare_experience,
    update_experience,
    collect_experience,
    incorporate_feedback,
    run_training_step,
    monitor_progress,
    train,
    reinitialize

"""
# Description:
    - Trainer implements the algorithm for collecting experience on which 
        a learner should learn. It is also responsible for monitoring learning 
        progress. In particular, a trainer should implement the following:
            + incorporate_feedback
            + update_experience
            + finalize_experience
            + monitor_progress
"""
abstract Trainer

finished_training(trainer::Trainer) = (
    trainer.step_count > trainer.max_step_count 
    || !has_time_remaining(trainer.monitor.timer)
    )
prepare_experience(trainer::Trainer, env::Env, policy::Policy) = trainer.experience

function update_experience(trainer::Trainer, x::Array{Float64}, 
        a::Array{Float64}, r::Union{Array{Float64},Float64}, nx::Array{Float64}, 
        done::Bool)
    trainer.step_count += 1
    update_experience(trainer.experience, x, a, r, nx, done)
end

function collect_experience(trainer::Trainer, env::Env, policy::Policy)
    reset_experience(trainer.experience)
    ix = reset!(env, trainer.initial_state_dist)
    for run in 1:trainer.num_mc_runs
        x = ix[:]
        reset!(env, x)
        done = false
        for t in 1:trainer.max_episode_steps
            a = step(policy, x)
            nx, r, done = step(env, a)
            update_experience(trainer, x, a, r, nx, done)
            x = nx
            if done break end
        end
    end
end

# base trainer does nothing with feedback
incorporate_feedback(trainer::Trainer, feedback::Dict{String, Array{Float64}}, 
    env::Env, policy::Policy) = trainer

function run_training_step(trainer::Trainer, learner::Learner, env::Env, policy::Policy)
    collect_experience(trainer, env, policy)
    feedback = learn(learner, prepare_experience(trainer, env, policy))
    incorporate_feedback(trainer, feedback, env, policy)
    monitor_progress(trainer.monitor, learner, trainer.step_count)
end

function train(trainer::Trainer, learner::Learner, env::Env, policy::Policy)
    while !finished_training(trainer)
        run_training_step(trainer, learner, env, policy)
    end
end

function reinitialize(trainer::Trainer)
    trainer.step_count = 0
    reset_experience(trainer.experience)
    reinitialize(trainer.monitor)
end

type AdaptiveTrainer <: Trainer
    initial_state_dist::Distribution
    monitor::TrainingMonitor
    update_dist_freq::Int
    n_components::Int
    feedback::Dict{String, Array{Float64}}
    max_episode_steps::Int
    num_mc_runs::Int
    experience::ExperienceMemory
    max_step_count::Int
    step_count::Int
    function AdaptiveTrainer(initial_state_dist::Distribution,
            monitor::TrainingMonitor;
            update_dist_freq::Int = 1000,
            n_components::Int = 2,
            max_episode_steps::Int = 1,
            num_mc_runs::Int = 1,
            max_step_count::Int = typemax(Int))
        return new(initial_state_dist, monitor, update_dist_freq, n_components,
            Dict{String, Array{Float64}}(), max_episode_steps, num_mc_runs,
            reset_experience(), max_step_count, 0)
    end
end

function incorporate_feedback(trainer::AdaptiveTrainer, 
        feedback::Dict{String, Array{Float64}}, env::Env, policy::Policy)
    update_feedback(trainer.feedback, feedback)
    if feedback_length(trainer.feedback) >= trainer.update_dist_freq
        # compute the utility weight of the samples as their softmax
        util_w = normalize_log_probs(trainer.feedback["errors"])

        # compute the likelihood ratio of each state under the environment 
        # sampling probability and the proposal distribution
        p_proposal = pdf(trainer.initial_state_dist, trainer.feedback["states"])
        p_env = pdf(env, trainer.feedback["states"])
        likelihood_w = p_env ./ p_proposal
        x_w = util_w .* reshape(likelihood_w, 1, length(likelihood_w))
        x_w ./= maximum(x_w)

        # refit the initial state distribution
        trainer.initial_state_dist = fit(typeof(trainer.initial_state_dist), 
            trainer.feedback["states"], 
            x_w = x_w, 
            n_components = trainer.n_components)

        # pause(trainer.monitor.timer)
        # a = plot_1d_dist(trainer.initial_state_dist)
        # save("/Users/wulfebw/Desktop/temp_risk/dist_$(trainer.step_count).pdf", a)
        # unpause(trainer.monitor.timer)

        # empty the trainers feedback
        empty!(trainer.feedback)
    end
end
