export
    plot_state_values,
    plot_learning_curve,
    plot_1d_dist

function plot_state_values(learner::Learner, monitor::TrainingMonitor,
        output_filepath::String)
    eval_states = monitor.eval_states
    n_eval_bins = length(eval_states)
    states = reshape(eval_states, n_eval_bins)
    values = reshape(predict(learner, eval_states), n_eval_bins)
    p = Plots.Linear(states, values)
    PGFPlots.save(output_filepath, p)
end

function plot_learning_curve(monitor::TrainingMonitor, output_filepath::String)
    curve = monitor.info["state-value rmse loss"]
    a = Axis(
        legendPos="south west", 
        width="16cm", 
        height="16cm", 
        xlabel="Training Time", 
        ylabel="Root Mean Square Error", 
        title="Error in Estimated Risk", 
        ymin=0., 
        ymax=1.2
    )
    p = Plots.Linear(collect(1:length(curve)), curve)
    push!(a, p)
    PGFPlots.save(output_filepath, a)
end

function plot_1d_dist(d::Distribution, n_samples::Int = 2000)
    samples = rand(d, n_samples)
    lo, hi = minimum(samples), maximum(samples)
    samples = reshape(samples, length(samples))
    a = Axis(Plots.Histogram(samples, bins=50), 
                xmin=lo, 
                xmax=hi, 
                width="8cm", 
                height="8cm",
                legendPos="north west"
    )
    return a
end

function plot_2d_dist(d::Distribution, n_samples::Int = 2000)
    samples = rand(d, n_samples)
    a = Axis(Plots.Scatter(samples[1,:], samples[2,:]), 
                width="8cm", 
                height="8cm"
    )
    return a
end