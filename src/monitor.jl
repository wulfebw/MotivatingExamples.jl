export
    TrainingMonitor,
    monitor_progress,
    log_value,
    is_on,
    reinitialize,
    restart

@with_kw type TrainingMonitor
    timer::BudgetTimer = BudgetTimer(typemax(Float64))
    eval_timer::BudgetTimer = BudgetTimer(typemax(Float64))
    eval_states::Array{Float64} = Float64[]
    v_true::Array{Float64} = Float64[]
    info::Dict{String, Any} = Dict{String, Any}()
    last_eval_step_count::Int = 0
    run_eval_every::Int = typemax(Int)
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
    monitor.last_eval_step_count = 0
    restart(monitor.eval_timer)
    restart(monitor.timer)
end

function should_run_eval(monitor::TrainingMonitor, step_count::Int)
    should_run = is_on(monitor)
    should_run &= (!has_time_remaining(monitor.eval_timer)
        || step_count - monitor.last_eval_step_count >= monitor.run_eval_every)

    # if true, then update tracking information
    if should_run
        restart(monitor.eval_timer)
        monitor.last_eval_step_count = step_count
    end
    
    return should_run
end

function run_eval(monitor::TrainingMonitor, learner::Learner)
    v_pred = predict(learner, monitor.eval_states)
    loss = rmse(monitor.v_true, v_pred)
    log_scalar(monitor, "state-value rmse loss", loss)
end

function monitor_progress(monitor::TrainingMonitor, learner::Learner,
        step_count::Int)
    pause(monitor.timer)
    if should_run_eval(monitor, step_count)
        run_eval(monitor, learner)
    end
    unpause(monitor.timer)
end
    