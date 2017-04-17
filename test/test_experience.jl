
function test_experience()
    mem = reset_experience()
    @test setdiff(Set(keys(mem)), Set(["x","a","r","nx","done"])) == Set{String}()
    @test length(mem) == 0
    update_experience(mem, [1.], [1.5], 3., [2.], true)
    @test length(mem) == 1
    @test mem["r"] == [[3.]]
    @test mem["x"] == [[1.]]
    x, a, r, nx, done = get(mem, 1)
    @test x == [1.]
    @test a == [1.5]
    @test r == [3.]
    @test nx == [2.]
    @test done == true
end

@time test_experience()