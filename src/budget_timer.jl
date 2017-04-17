export
    BudgetTimer,
    has_time_remaining,
    pause,
    unpause,
    past_fraction

type BudgetTimer
    budget::Float64 # budget in seconds
    start_time::Float64
    pause_start::Float64
    function BudgetTimer(budget::Float64)
        return new(budget, time(), 0.)
    end
end

function has_time_remaining(timer::BudgetTimer)
    return (time() - timer.start_time) < timer.budget
end

function pause(timer::BudgetTimer)
    timer.pause_start = time()
end

function unpause(timer::BudgetTimer)
    timer.start_time += time() - timer.pause_start
end

function restart(timer::BudgetTimer) 
    timer.start_time = time()
end