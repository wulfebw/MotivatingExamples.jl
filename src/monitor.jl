export
    TrainingMonitor,
    monitor_progress,
    log_value,
    is_on

@with_kw type TrainingMonitor
    timer::BudgetTimer = BudgetTimer(10.)
    eval_timer::BudgetTimer = BudgetTimer(1.)
    eval_states::Array{Float64} = Float64[]
    v_true::Array{Float64} = Float64[]
    info::Dict{String, Any} = Dict{String, Any}()
end

is_on(monitor::TrainingMonitor) = return length(monitor.v_true) > 0
log_value(monitor::TrainingMonitor, k::String, v::Any) = monitor.info[k] = v

function monitor_progress(monitor::TrainingMonitor, learner::Learner)
    pause(monitor.timer)
    if is_on(monitor) && !has_time_remaining(monitor.eval_timer)
        v_pred = predict(learner, monitor.eval_states)
        loss = rmse(monitor.v_true, v_pred)
        log_value(monitor, "state-value rmse loss", loss)
        restart(monitor.eval_timer)
    end
    unpause(monitor.timer)
end
    