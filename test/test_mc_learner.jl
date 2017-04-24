# using Base.Test
# using MotivatingExamples

function build_debug_mc_learner()
    minpos = 0
    maxpos = 1
    nbins = 2
    cutpoints = linspace(minpos, maxpos, nbins)
    grid = RectangleGrid(cutpoints)
    target_dim = 1
    discount = .5
    lr = .1
    learner = MCLearner(grid, target_dim, discount = discount, lr = lr)
    return learner
end

function test_compute_single_episode_state_returns()
    mem = reset_experience()
    learner = build_debug_mc_learner()
    discount = .5
    update_experience(mem, [0.], [1.], [1.], [1.], false)
    update_experience(mem, [1.], [1.], [5.], [2.], false)
    update_experience(mem, [2.], [1.], [10.], [0.], true)
    state_returns = compute_single_episode_state_returns(mem, learner, discount)
    
    states = [[0.], [1.], [2.]]
    rets = [[.5^2 * 10. + .5^1 * 5. + 1.], [.5 * 10. + 5.], [10.]]

    state_returns = reverse(state_returns)
    for (expect_x, expect_ret, (actual_x, actual_ret)) in zip(states, rets, state_returns)
        @test all(expect_x .== actual_x)
        @test all(expect_ret .== actual_ret)
    end

    mem = reset_experience()
    learner.values = ones(size(learner.values))
    update_experience(mem, [0.], [1.], [1.], [1.], false)
    update_experience(mem, [1.], [1.], [5.], [2.], false)
    state_returns = compute_single_episode_state_returns(mem, learner, discount)

    states = [[0.], [1.]]
    rets = [[.5 * 5.5 + 1.], [5. + .5 * 1.]]

    state_returns = reverse(state_returns)
    for (expect_x, expect_ret, (actual_x, actual_ret)) in zip(states, rets, state_returns)
        @test all(expect_x .== actual_x)
        @test all(expect_ret .== actual_ret)
    end

end

function test_simple_mc_learning()
    srand(0)
    minpos = 0
    maxpos = 1
    nbins = 2
    cutpoints = linspace(minpos, maxpos, nbins)
    grid = RectangleGrid(cutpoints)
    target_dim = 1
    discount = .5
    lr = .1
    learner = MCLearner(grid, target_dim, discount = discount, lr = lr)

    mem = reset_experience()
    update_experience(mem, [0.], [1.], [1.], [1.], false)
    update_experience(mem, [1.], [1.], [5.], [1.], true)
    learn(learner, mem)
    @test learner.values[1] ≈ .1 + .1 * .5 * 5
    @test learner.values[2] == .5
end

@time test_compute_single_episode_state_returns()
@time test_simple_mc_learning()
