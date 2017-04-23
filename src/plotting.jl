export
    plot_state_values,
    plot_learning_curve

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
        xlabel="Seconds", 
        ylabel="Root Mean Square Error", 
        title="Error in Estimated Risk", 
        ymin=0., 
        ymax=.6
    )
    p = Plots.Linear(collect(1:length(curve)), curve)
    push!(a, p)
    PGFPlots.save(output_filepath, a)
end