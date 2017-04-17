
function test_univariate_gaussian_policy()
    pi = UnivariateGaussianPolicy()
    a = step(pi, [0.])
    @test length(a) == 1
end

@time test_univariate_gaussian_policy()