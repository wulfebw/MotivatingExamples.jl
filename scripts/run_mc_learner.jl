using MotivatingExamples
using JLD
using PGFPlots

function update_results(results::Dict, trainer::Trainer, learner::Learner, 
        env::Env, policy::Policy, key::Any)
    monitor = trainer.monitor
    results[key] = Dict()
    results[key]["learning_curve"] = monitor.info["state-value rmse loss"][:]
    state_values = predict(learner, monitor.eval_states)
    results[key]["state_values"] = state_values
    results[key]["loss"] = rmse(monitor.v_true, state_values)
end

# env
xmin = -100000.
xmax = 100000.
env = Continuous1DRandomWalkEnv(xmin = xmin, xmax = xmax)

# policy
σ = 1.
policy = UnivariateGaussianPolicy(σ = σ)

# learner settings (built below)
nbins = 20
discount = 1.
lr = 0.05
bins = linspace(env.xmin, env.xmax, nbins)
grid = RectangleGrid(bins)
target_dim = 1

# trainer settings(built below)
# information tracking is done at regular environment step interval 
# as opposed to real-time intervals because the former seems to be more reliable
n_eval_bins = 100
n_eval_samples = 20
budget = typemax(Float64)
timer = BudgetTimer(budget)
max_step_count = 5000
run_eval_every = Int(ceil(max_step_count / n_eval_samples)) 
eval_states = reshape(linspace(env.xmin, env.xmax, n_eval_bins), (1, n_eval_bins))

# compute true state values
m = 1. / (env.xmax - env.xmin)
b = env.xmin
v_true = reshape(eval_states, 1, n_eval_bins) .* m .+ .5

# build the monitor
monitor = TrainingMonitor(timer = timer, eval_states = eval_states, 
    v_true = v_true, run_eval_every = run_eval_every)

# define the hyperparameter options


# run each hyperparameter setting and collect results
results = Dict()
runs_per_setting = 2

learner = MCLearner(grid, target_dim, discount = discount, lr = lr)
trainer = AdaptiveTrainer(env.initial_state_dist, monitor,
    max_episode_steps = 5,
    num_mc_runs = 1,
    max_step_count = max_step_count)
for run in 1:runs_per_setting
    println("run $(run) / $(runs_per_setting)")
    reinitialize(learner)
    reinitialize(trainer)
    train(trainer, learner, env, policy)
    update_results(results, trainer, learner, env, policy, run)
end


# accumulate results
learning_curve = zeros(n_eval_samples)
state_values = zeros(n_eval_bins)
loss = 0
min_n_eval_samples = typemax(Int)
for run in 1:runs_per_setting
    n_samples = length(results[run]["learning_curve"])
    min_n_eval_samples = min(n_samples, min_n_eval_samples)
    for k in 1:n_samples
        learning_curve[k] += results[run]["learning_curve"][k]
    end
    state_values += reshape(results[run]["state_values"], n_eval_bins)
    loss += results[run]["loss"]
end


# average across runs and truncate to a number of samples hit by all the 
# individual runs
learning_curve /= runs_per_setting
learning_curve = learning_curve[1:min_n_eval_samples]
state_values /= runs_per_setting

# plot
a = Axis(legendPos="south west", width="16cm", height="16cm", xlabel="Steps", 
        ylabel="Root Mean Square Error", title="Error in Estimated Risk", ymin=0., 
        ymax=.6)
p = Plots.Linear(collect(1:length(learning_curve)) .* run_eval_every, learning_curve[:,1], 
    legendentry="monte carlo",
    mark="diamond",
    style="orange,thick",
    markSize=4)
push!(a, p)
PGFPlots.save("../data/visualizations/mc_learning_curves.pdf", a)

# the value function
p = Axis(Plots.Linear(eval_states[1,:], state_values, 
    legendentry="error: $(round(loss, 3))"), 
    width="5.5cm", 
    height="5.5cm",
    legendPos="north west",
    title=string("monte carlo"))
PGFPlots.save("../data/visualizations/mc_state_values.pdf", p)
