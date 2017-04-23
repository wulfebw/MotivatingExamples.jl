
function test_univariate_gaussian_policy()
    pi = UnivariateGaussianPolicy()
    srand(pi, 0)
    a = step(pi, [0.])
    @test length(a) == 1

    srand(pi, 0)
    b = step(pi, [0.])
    @test all(a .== b)
end

@time test_univariate_gaussian_policy()