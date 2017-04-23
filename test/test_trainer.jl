# using Base.Test
# using MotivatingExamples
# using PGFPlots

function build_debug_setup(;
        # general hyperparams
        discount::Float64 = 1.,
        lr::Float64 = .01,
        budget::Float64 = 5.,
        n_eval_bins::Int = 100,
        max_episode_steps::Int = 5,
        num_mc_runs::Int = 1,
        max_step_count::Int = typemax(Int),
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
    nbins = 5
    bins = linspace(env.xmin, env.xmax, nbins)
    grid = RectangleGrid(bins)
    target_dim = 1
    learner = TDLearner(grid, target_dim, discount = discount, lr = lr)

    # trainer
    initial_state_dist = env.initial_state_dist
    eval_states = reshape(linspace(env.xmin, env.xmax, n_eval_bins), (1, n_eval_bins))
    m = 1. / (env.xmax - env.xmin)
    b = env.xmin
    v_true = reshape(eval_states, 1, n_eval_bins) .* m .+ .5
    timer = BudgetTimer(budget)
    monitor = TrainingMonitor(timer = timer, eval_states = eval_states, 
        v_true = v_true)
    max_episode_steps = max_episode_steps
    num_mc_runs = num_mc_runs
    trainer = AdaptiveTrainer(initial_state_dist, monitor,
        max_episode_steps = max_episode_steps,
        num_mc_runs = num_mc_runs,
        max_step_count = max_step_count)

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

@time test_trainer()
@time test_trainer_reinitialize()