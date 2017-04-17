export 
    ExperienceType,
    ExperienceMemory,
    length,
    reset_experience,
    update_experience,
    get

typealias ExperienceType Union{Array{Float64}, Float64, Bool}
typealias ExperienceMemory Dict{String,Array{ExperienceType}}

length(mem::ExperienceMemory) = length(mem["x"])

function reset_experience(d::ExperienceMemory = ExperienceMemory())
    for k in ["x", "a", "r", "nx", "done"]
        d[k] = ExperienceType[]
    end
    return d
end

function update_experience(experience::ExperienceMemory, x::Array{Float64}, 
        a::Array{Float64}, r::Union{Float64,Array{Float64}}, nx::Array{Float64}, 
        done::Bool)
    push!(experience["x"], x)
    push!(experience["a"], a)
    if typeof(r) == Float64 
        r = [r] 
    end
    push!(experience["r"], r)
    push!(experience["nx"], nx)
    push!(experience["done"], done)
end

function get(experience::ExperienceMemory, index::Int)
    @assert index <= length(experience)
    x = experience["x"][index]
    a = experience["a"][index]
    r = experience["r"][index]
    nx = experience["nx"][index]
    done = experience["done"][index]
    return x, a, r, nx, done
end