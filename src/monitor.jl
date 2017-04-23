export
    TrainingMonitor,
    monitor_progress,
    log_value,
    is_on,
    reinitialize,
    restart

@with_kw type TrainingMonitor
    timer::BudgetTimer = BudgetTimer(10.)
    eval_timer::BudgetTimer = BudgetTimer(1.)
    eval_states::Array{Float64} = Float64[]
    v_true::Array{Float64} = Float64[]
    info::Dict{String, Any} = Dict{String, Any}()
end

is_on(monitor::TrainingMonitor) = return length(monitor.v_true) > 0
function log_scalar(monitor::TrainingMonitor, k::String, v::Any) 
    if in(k, keys(monitor.info))
        push!(monitor.info[k], v)
    else
        t = typeof(v)
        monitor.info[k] = t[]
    end
end

# only restarts the training time
restart(monitor::TrainingMonitor) = restart(monitor.timer)

# restarts training time, eval time, and info
function reinitialize(monitor::TrainingMonitor) 
    monitor.info = Dict{String, Any}()
    restart(monitor.eval_timer)
    restart(monitor.timer)
end

function monitor_progress(monitor::TrainingMonitor, learner::Learner)
    pause(monitor.timer)
    if is_on(monitor) && !has_time_remaining(monitor.eval_timer)
        v_pred = predict(learner, monitor.eval_states)
        loss = rmse(monitor.v_true, v_pred)
        log_scalar(monitor, "state-value rmse loss", loss)
        restart(monitor.eval_timer)
    end
    unpause(monitor.timer)
end
    