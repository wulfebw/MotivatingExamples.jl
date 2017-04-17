
function test_trainer()
    srand(0)
    # env
    env = Continuous1DRandomWalkEnv()

    # policy
    policy = UnivariateGaussianPolicy()

    # learner
    nbins = 5
    bins = linspace(env.xmin, env.xmax, nbins)
    grid = RectangleGrid(bins)
    target_dim = 1
    learner = TDLearner(grid, target_dim, discount = 1., lr = .01)

    # trainer
    initial_state_dist = env.initial_state_dist
    budget = 5.
    n_eval_bins = 100
    eval_states = reshape(linspace(env.xmin, env.xmax, n_eval_bins), (1, n_eval_bins))
    m = 1. / (env.xmax - env.xmin)
    b = env.xmin
    v_true = reshape(eval_states, 1, n_eval_bins) .* m .+ .5
    timer = BudgetTimer(budget)
    monitor = TrainingMonitor(timer = timer, eval_states = eval_states, 
        v_true = v_true)
    max_episode_steps = 5
    num_mc_runs = 1
    trainer = AdaptiveTrainer(initial_state_dist, monitor,
        max_episode_steps = max_episode_steps,
        num_mc_runs = num_mc_runs)

    # run training
    train(trainer, learner, env, policy)

    loss = rmse(monitor.v_true, predict(learner, monitor.eval_states))
    @test loss < .1
    
    # plot it
    # states = reshape(monitor.eval_states, n_eval_bins)
    # values = reshape(predict(learner, monitor.eval_states), n_eval_bins)
    # p = Plots.Linear(states, values)
    # PGFPlots.save("../data/trainer_test.pdf", p)

end

@time test_trainer()