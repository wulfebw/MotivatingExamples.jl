export
    GaussianMixtureModel,
    get_gmm_dists,
    initialize_gmm,
    compute_gmm_ll,
    fit_gmm,
    fit,
    em,
    rand,
    pdf,
    logpdf

function get_gmm_dists(μ::Array{Float64}, Σ::Array{Float64}, K::Int)
    dists = MvNormal[]
    try
        dists = [MvNormal(μ[:,k], Σ[:,:,k]) for k in 1:K]
    catch e
        println("exception raised while getting the gmm dists: $(e)")
        for i in 1:K
            println("component $(i)")
            println("μ: $(μ[:,i])")
            println("Σ: $(Σ[:,:,i])")
        end
        throw(e)
    end
    return dists
end

type GaussianMixtureModel <: Distribution
    m::Multinomial
    dists::Vector{MvNormal}
    function GaussianMixtureModel(π::Array{Float64}, μ::Array{Float64}, 
        Σ::Array{Float64})
        return new(Multinomial(1, π), get_gmm_dists(μ, Σ, length(π)))
    end
end

function rand(d::GaussianMixtureModel)
    c = indmax(rand(d.m))
    return rand(d.dists[c])
end
rand(rng::MersenneTwister, d::GaussianMixtureModel) = rand(d)
function rand(d::GaussianMixtureModel, n_samples::Int)
    s = rand(d)
    dim = length(s)
    samples = zeros(dim, n_samples)
    samples[:, 1] = s
    for i in 1:(n_samples - 1)
        samples[:, i] = rand(d)
    end
    return samples
end

function logpdf(d::GaussianMixtureModel, x::Array{Float64})
    prob = 0 
    for (i, mvd) in enumerate(d.dists)
        prob += d.m.p[i] * pdf(mvd, x)
    end
    return log(prob)
end
pdf(d::GaussianMixtureModel, x::Array{Float64}) = exp(logpdf(d, x))

#=
Fitting
=#

function initialize_gmm(K::Int, x::Array{Float64}, 
        x_w::Array{Float64} = ones(size(x, 2)))
    D, N = size(x)
    w = zeros(K, N)

    # means 
    μ = zeros(D, K)
    step = Int(ceil(D / K))
    for k in 1:K
        s = (k-1) * step
        e = s + step
        μ[:,k] = sum(x[:, s+1:e] .* x_w) / sum(x_w)
    end

    # covariance matrices
    Σ = zeros(D, D, K)
    for k in 1:K
        Σ[:,:,k] = eye(D)
    end

    # class prior weights
    π = rand(K)
    π ./= sum(π)

    return w, μ, Σ, π
end

function compute_gmm_ll(π::Array{Float64}, μ::Array{Float64}, Σ::Array{Float64}, 
        x::Array{Float64}, x_w::Array{Float64} = ones(size(x, 2)))
    N, K = size(x, 2), length(π)
    dists = get_gmm_dists(μ, Σ, K)
    ll = 0
    for sidx in 1:N
        total = 0
        for k in 1:K
            total += π[k] * pdf(dists[k], x[:, sidx])
        end
        ll += x_w[sidx] * log(total)
    end
    return ll
end

function fit_gmm(x::Array{Float64}; 
        x_w::Array{Float64} = ones(1, size(x, 2)), 
        max_iters::Int = 30, 
        tol::Float64 = 1e-2, 
        n_components::Int = 2,
        verbose::Bool = false)

    # initialize
    D, N = size(x)
    K = n_components
    w, μ, Σ, π = initialize_gmm(K, x, x_w)
    dists = get_gmm_dists(μ, Σ, K)

    prev_ll = compute_gmm_ll(π, μ, Σ, x, x_w)
    for iteration in 1:max_iters
        
        # e-step
        log_π = log(π)
        for sidx in 1:N
            for k in 1:K
                w[k, sidx] = log_π[k] + logpdf(dists[k], x[:, sidx])
            end
        end
        w = normalize_log_probs(w)
        # account for sample probability after normalizing because 
        # the sample weights do not impact normalization and this is more 
        # efficient
        w .*= x_w 

        # m-step
        π = sum(w, 2) ./ sum(w)
        
        μ = zeros(D, K)
        for k in 1:K
            for sidx in 1:N
                μ[:, k] += w[k,sidx] * x[:,sidx]
            end
            μ[:, k] ./= sum(w[k,:])
        end
        
        Σ = ones(D,D,K) * 1e-8
        for k in 1:K
            for sidx in 1:N
                diff = x[:,sidx] - μ[:, k]
                Σ[:, :, k] += w[k, sidx] * (diff * transpose(diff))
            end
            Σ[:, :, k] ./= sum(w[k, :])
        end
        
        # check for convergence
        ll = compute_gmm_ll(π, μ, Σ, x, x_w)
        if verbose
            println("iteration: $(iteration) / $(max_iters)\t ll: $(ll)")
        end
        if abs(ll - prev_ll) < tol
            break
        else
            prev_ll = ll
            dists = get_gmm_dists(μ, Σ, K)
        end
    end
    return π, μ, Σ 
end

function fit(T::Type{GaussianMixtureModel},
        x::Array{Float64}; 
        x_w::Array{Float64} = ones(1, size(x, 2)), 
        max_iters::Int = 30, 
        tol::Float64 = 1e-2, 
        n_components::Int = 2)
    π, μ, Σ = fit_gmm(x, x_w = x_w, max_iters = max_iters, tol = tol, 
        n_components = n_components)
    return GaussianMixtureModel(π[:], μ, Σ)
end