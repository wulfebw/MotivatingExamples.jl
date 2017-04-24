using MotivatingExamples
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
xmin = -100.
xmax = 100.
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
max_step_count = 100000
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
max_episode_steps_options = [1, 5, 20]
n_ep_opts = length(max_episode_steps_options)
num_mc_runs_options = [1, 2, 5]
n_mc_opts =length(num_mc_runs_options)

# run each hyperparameter setting and collect results
results = Dict()
runs_per_setting = 2
for (i, max_episode_steps) in enumerate(max_episode_steps_options)
    println("ep steps $(i) / $(n_ep_opts)")
    for (j, num_mc_runs) in enumerate(num_mc_runs_options)
        println("mc $(j) / $(n_mc_opts)")
        learner = TDLearner(grid, target_dim, discount = discount, lr = lr)
        trainer = AdaptiveTrainer(env.initial_state_dist, monitor,
            max_episode_steps = max_episode_steps,
            num_mc_runs = num_mc_runs,
            max_step_count = max_step_count)
        for run in 1:runs_per_setting
            println("run $(run) / $(runs_per_setting)")
            reinitialize(learner)
            reinitialize(trainer)
            train(trainer, learner, env, policy)
            key = (max_episode_steps, num_mc_runs, run)
            update_results(results, trainer, learner, env, policy, key)
        end
    end
end

# plot results
learnings_curves = zeros(n_eval_samples, n_mc_opts, n_ep_opts)
state_values = zeros(n_eval_bins, n_mc_opts, n_ep_opts)
losses = zeros(n_mc_opts, n_ep_opts)
min_n_eval_samples = typemax(Int)
for (i, max_episode_steps) in enumerate(max_episode_steps_options)
    for (j, num_mc_runs) in enumerate(num_mc_runs_options)
        for run in 1:runs_per_setting
            key = (max_episode_steps, num_mc_runs, run)
            n_samples = length(results[key]["learning_curve"])
            min_n_eval_samples = min(n_samples, min_n_eval_samples)
            for k in 1:n_samples
                learnings_curves[k, i, j] += results[key]["learning_curve"][k]
            end
            state_values[:, i, j] += reshape(results[key]["state_values"], n_eval_bins)
            losses[i, j] += results[key]["loss"]
        end
    end
end

# average across runs and truncate to a number of samples hit by all the 
# individual runs
learnings_curves /= runs_per_setting
learnings_curves = learnings_curves[1:min_n_eval_samples, :, :]
state_values /= runs_per_setting

# plot
a = Axis(legendPos="south west", width="16cm", height="16cm", xlabel="Seconds", 
    ylabel="Root Mean Square Error", title="Error in Estimated Risk", ymin=0., 
    ymax=.6)
markcolors = ["red", "blue", "green", "black"]
markers = ["square", "triangle", "o", "diamond"]

for (i, max_episode_steps) in enumerate(max_episode_steps_options)
    markcolor = markcolors[i]
    for (j, num_mc_runs) in enumerate(num_mc_runs_options)
        marker = markers[j]
        avg_curve = learnings_curves[:,i,j]
        p = Plots.Linear(1:length(avg_curve), avg_curve[:,1], 
        legendentry="steps: $(max_episode_steps) monte carlo runs: $(num_mc_runs)",
        mark="$(marker)",
        style="$(markcolor),thick",
        markSize=4)
        push!(a, p)
    end
end
PGFPlots.save("../data/visualizations/learning_curves.pdflearning_curves.pdf", a)


# plot all the value functions
g = GroupPlot(n_ep_opts, n_mc_opts, 
    groupStyle = "horizontal sep = 2cm, vertical sep = 2cm")
for (i, max_episode_steps) in enumerate(max_episode_steps_options)
    for (j, num_mc_runs) in enumerate(num_mc_runs_options)
        p = Axis(Plots.Linear(eval_states[1,:], state_values[:,i,j], 
            legendentry="error: $(round(losses[i,j], 3))"), 
            width="5.5cm", 
            height="5.5cm",
        legendPos="north west",
        title=string("steps: $(max_episode_steps) monte carlo runs: $(num_mc_runs)"))
        push!(g, p)
    end
end
PGFPlots.save("../data/visualizations/state_values.pdf", g)


