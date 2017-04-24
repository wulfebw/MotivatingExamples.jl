# using Base.Test
# using MotivatingExamples

function test_simple_td_learning()
    srand(0)
    minpos = 0
    maxpos = 1
    nbins = 6
    bins = linspace(minpos, maxpos, nbins)
    grid = RectangleGrid(bins, bins)
    target_dim = 2
    learner = TDLearner(grid, target_dim, discount = 0.5)

    # generate dataset
    # dataset consists of 3 dimensional states
    # where the last state is irrelevant
    # and the values are noisy functions of x and y
    num_samples = 1000
    eps = 1e-2
    mem = reset_experience()
    for i in 1:num_samples
        x = rand(minpos:eps:maxpos)
        y = rand(minpos:eps:maxpos)
        a = eps * randn()
        r = [1, x + y]
        nx = [x + a, y + a]
        update_experience(mem, [x,y], [a], r, nx, false)
    end
    
    # fit the model
    num_epochs = 100
    for epoch in 1:num_epochs
        learn(learner, mem)
    end

    # visualize
    num_steps = 10
    steps = linspace(minpos, maxpos, num_steps)
    v = zeros(target_dim, num_steps, num_steps)
    for (i, x) in enumerate(steps)
        for (j, y) in enumerate(steps)
            v[:,i,j] = predict(learner, [x, y])
        end
    end

    @test all(abs(v[1,:] .- 2) .< 1e-4)
    @test predict(learner, [.25,.25])[2] < predict(learner, [.75,.75])[2]

    # # visualize it
    # function get_heat_1(x, y)
    #     predict(learner, [x, y])[1]
    # end
    # function get_heat_2(x, y)
    #     predict(learner, [x, y])[2]
    # end

    # nbins = 50
    # img = Plots.Image(get_heat_1, (minpos, maxpos), (minpos, maxpos), 
    #     xbins=nbins, ybins=nbins)
    # PGFPlots.save("../data/test1.pdf", img)
    # img = Plots.Image(get_heat_2, (minpos, maxpos), (minpos, maxpos), 
    #     xbins=nbins, ybins=nbins)
    # PGFPlots.save("../data/test2.pdf", img)
end

function test_learner_reinitialize()
    srand(0)
    minpos = 0
    maxpos = 1
    nbins = 6
    bins = linspace(minpos, maxpos, nbins)
    grid = RectangleGrid(bins, bins)
    target_dim = 2
    learner = TDLearner(grid, target_dim, discount = 0.5)

    mem = reset_experience()
    update_experience(mem, [1.,2.], [1.], [1.,2.], [2.,1.], false)
    learn(learner, mem)
    values1 = learner.values[:]
    reinitialize(learner)
    @test all(learner.values .== zeros(target_dim, length(grid)))
    learn(learner, mem)
    values2 = learner.values[:]
    @test all(values1 .== values2)
end

@time test_simple_td_learning()
@time test_learner_reinitialize()
