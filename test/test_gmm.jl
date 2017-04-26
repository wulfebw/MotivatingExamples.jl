using Base.Test
using JLD
using MotivatingExamples
using PGFPlots

function test_gmm_on_test_data()
    input_filepath = "../data/bug_states.jld"
    f = JLD.load(input_filepath)
    x = f["states"]
    x_w = f["x_w"]
    s = Plots.Scatter(x[1,:], x[2,:])
    PGFPlots.save("/Users/wulfebw/Desktop/gmm.pdf", s)

    π, μ, Σ = fit_gmm(x, x_w = x_w)
end

@time test_gmm_on_test_data()