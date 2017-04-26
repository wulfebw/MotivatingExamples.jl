using Base.Test
using JLD
using MotivatingExamples
using PGFPlots

function build_debug_setup(;
        # general hyperparams
        discount::Float64 = 1.,
        lr::Float64 = .05,
        budget::Float64 = 5.,
        n_eval_bins::Int = 100,
        max_episode_steps::Int = 5,
        num_mc_runs::Int = 1,
        max_step_count::Int = typemax(Int),
        update_dist_freq = typemax(Int),
        initial_state_dist = nothing,
        run_eval_every = typemax(Int),
        # learner
        nbins = 10,
        # env
        xmin::Float64 = -10.,
        xmax::Float64 = 10.,
        # policy
        σ::Float64 = 1.
        )
    # env
    env = Continuous1DRandomWalkEnv(xmin = xmin, xmax = xmax)

    # policy
    policy = UnivariateGaussianPolicy(σ = σ)

    # learner
    bins = linspace(env.xmin, env.xmax, nbins)
    grid = RectangleGrid(bins)
    target_dim = 1
    learner = TDLearner(grid, target_dim, discount = discount, lr = lr)

    # trainer
    if initial_state_dist == nothing
        initial_state_dist = env.initial_state_dist
    end
    eval_states = reshape(linspace(env.xmin, env.xmax, n_eval_bins), (1, n_eval_bins))
    m = 1. / (env.xmax - env.xmin)
    b = env.xmin
    v_true = reshape(eval_states, 1, n_eval_bins) .* m .+ .5
    timer = BudgetTimer(budget)
    monitor = TrainingMonitor(timer = timer, eval_states = eval_states, 
        v_true = v_true, run_eval_every = run_eval_every)
    trainer = AdaptiveTrainer(initial_state_dist, monitor,
        max_episode_steps = max_episode_steps,
        num_mc_runs = num_mc_runs,
        max_step_count = max_step_count,
        update_dist_freq = update_dist_freq)

    return trainer, learner, env, policy
end

function test_trainer()
    srand(0)
    
    # build everything
    trainer, learner, env, policy = build_debug_setup()
    monitor = trainer.monitor
    n_eval_bins = length(monitor.eval_states)

    # run training
    train(trainer, learner, env, policy)

    loss = rmse(monitor.v_true, predict(learner, monitor.eval_states))
    @test loss < .1
    
    # plotting
    # plot_state_values(learner, monitor, "../data/visualizations/test_state_values.pdf")
    # plot_learning_curve(monitor, "../data/visualizations/test_learning_curve.pdf")
end

function test_trainer_reinitialize()
    budget = 10.
    max_step_count = 1000
    srand(0)
    trainer, learner, env, policy = build_debug_setup(
        max_step_count = max_step_count, budget = budget)
    monitor = trainer.monitor
    srand(env.rng, 1)
    srand(policy.rng, 1)
    reinitialize(monitor)
    train(trainer, learner, env, policy)
    loss_1 = rmse(monitor.v_true, predict(learner, monitor.eval_states))
    
    srand(0)
    trainer, learner, env, policy = build_debug_setup(
        max_step_count = max_step_count, budget = budget)
    monitor = trainer.monitor
    srand(env.rng, 1)
    srand(policy.rng, 1)
    train(trainer, learner, env, policy)
    loss_2 = rmse(monitor.v_true, predict(learner, monitor.eval_states))

    @test loss_1 == loss_2
    
    reinitialize(learner)
    reinitialize(trainer)
    srand(env, 1)
    srand(policy, 1)
    train(trainer, learner, env, policy)
    loss_3 = rmse(monitor.v_true, predict(learner, monitor.eval_states))
    @test loss_1 == loss_3
end

function test_adaptive_trainer_1d_env()
     srand(0)
    
    # build everything
    budget = 10.
    run_eval_every = 1000
    xmin = -10.
    xmax = 10.
    π = [.49, .51]
    μ = reshape([xmin / 2, xmax / 2], 1, 2)

    σ = ((xmax - xmin) / 2.)^2
    Σ = reshape([σ, σ], 1, 1, 2)
    dist = GaussianMixtureModel(π, μ, Σ)

    trainer, learner, env, policy = build_debug_setup(
        initial_state_dist = dist,
        update_dist_freq = 100,
        max_episode_steps = 1,
        n_eval_bins = 1000,
        xmin = xmin,
        xmax = xmax,
        budget = budget,
        run_eval_every = run_eval_every,
        lr = 0.1,
        nbins = 40)
    monitor = trainer.monitor
    n_eval_bins = length(monitor.eval_states)

    # run training
    train(trainer, learner, env, policy)

    v_pred = predict(learner, monitor.eval_states)
    loss = rmse(monitor.v_true, v_pred)
    @test loss < .2

    # plotting
    # plot_state_values(learner, monitor, 
    #     "/Users/wulfebw/Desktop/adaptive_test_state_values.pdf")
    # plot_learning_curve(monitor, 
    #     "/Users/wulfebw/Desktop/adaptive_test_learning_curve.pdf")

end

function test_adaptive_trainer_2d_env()
     srand(0)
    
    # env
    xmin = -100.
    xmax = 100.
    ymin = -100.
    ymax = 100.
    env = Continuous2DRareEventEnv(
        xmin = xmin, 
        xmax = xmax,
        ymin = ymin,
        ymax = ymax,
        max_thresh = 4.,
        min_thresh = 1.5)

    # policy
    policy = MultivariateGaussianPolicy()

    # learner
    nbins = 10
    discount = 1.
    lr = 0.1
    xbins = linspace(env.xmin, env.xmax, nbins)
    ybins = linspace(env.ymin, env.ymax, nbins)
    grid = RectangleGrid(xbins, ybins)
    target_dim = 1
    learner = TDLearner(grid, target_dim, discount = discount, lr = lr)

    # trainer
    budget = 2.
    run_eval_every = 5000
    ## dist
    π = [.5, .5]
    μ = zeros(2,2)
    μ[1,1] = xmin / 2
    μ[2,1] = ymin / 2
    μ[2,1] = xmax / 2
    μ[2,2] = ymax / 2
    σ = ((xmax - xmin) / 4.)^2
    Σ = reshape(hcat(eye(2),eye(2)), 2,2,2) .* σ
    dist = GaussianMixtureModel(π, μ, Σ)

    n_eval_bins = 40
    eval_states_x = linspace(env.xmin, env.xmax, n_eval_bins)
    eval_states_y = linspace(env.ymin, env.ymax, n_eval_bins)
    eval_states = zeros(2, n_eval_bins, n_eval_bins)
    for i in 1:n_eval_bins
        for j in 1:n_eval_bins
        eval_states[1,i,j] = eval_states_x[i]
        eval_states[2,i,j] = eval_states_y[j]
        end
    end
    eval_states = reshape(eval_states, 2, n_eval_bins * n_eval_bins)

    v_true = ones(n_eval_bins * n_eval_bins)
    monitor = TrainingMonitor(timer = BudgetTimer(budget), 
        eval_states = eval_states, v_true = v_true, 
        run_eval_every = run_eval_every)

    trainer = AdaptiveTrainer(dist, monitor,
        max_episode_steps = 1,
        num_mc_runs = 1,
        max_step_count = typemax(Int),
        update_dist_freq = 10000000)

    # run training
    train(trainer, learner, env, policy)

    v_pred = reshape(predict(learner, eval_states), size(v_true))
    final_mse = rmse(v_true, v_pred)

    # save everything
    output_filepath = "../data/uniform.jld"
    JLD.save(output_filepath, "learning_curve", monitor.info["state-value rmse loss"], "state_values", v_pred)

    # visualize it
    get_heat(x, y) = predict(learner, [x,y])[1]

    nbins = 20
    img = Axis(Plots.Image(get_heat, (env.xmin, env.xmax), (env.ymin, env.ymax),
        xbins=nbins, ybins=nbins), title="state-values; error: $(round(final_mse, 5))")
    PGFPlots.save("/Users/wulfebw/Desktop/test_2d.pdf", img)

    plot_learning_curve(monitor, 
        "/Users/wulfebw/Desktop/adaptive_test_learning_curve.pdf")
end

println("test_trainer.jl")
@time test_trainer()
@time test_trainer_reinitialize()
# @time test_adaptive_trainer_1d_env()
# @time test_adaptive_trainer_2d_env()