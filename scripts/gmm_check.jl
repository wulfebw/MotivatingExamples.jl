# check the EM implementation
g = GroupPlot(2, 2, groupStyle = "horizontal sep = 1.75cm, vertical sep = 1.5cm")
sc = "{a={mark=diamond,blue},b={mark=diamond,red},c={mark=diamond,green}}"

# 2d check
num_samples = 1000
samples = zeros(2, num_samples)
gauss1 = MvNormal([-2.,-2.],[[1., 0.] [0., 1.]])
gauss2 = MvNormal([2.,2.],[[1., .9] [.9, 1.]])
gauss3 = MvNormal([-2.,2.],[[1., -.9] [-.9, 1.]])
z = String[]
for i in 1:size(samples, 2)
    v = rand()
    if v < .33
        samples[:, i] = rand(gauss1)
        push!(z, "a")
    elseif v < .66
        samples[:, i] = rand(gauss2)
        push!(z, "b")
    else
        samples[:, i] = rand(gauss3)
        push!(z, "c")
    end
end
scatter_orig = Plots.Scatter(samples[1,:], samples[2,:], z, scatterClasses=sc)
push!(g, scatter_orig)

pis, mus, sigmas = fit_gmm(samples, num_components = 3)
println("pis: $(pis)")
println("mus: $(mus)")
println("sigmas: $(sigmas)")

N, K = size(samples, 2), length(pis)
dists = [MvNormal(mus[:,k], sigmas[:,:,k]) for k in 1:K]
z = String[]
for sidx in 1:N
    p1 = pis[1] * pdf(dists[1], samples[:, sidx])
    p2 = pis[2] * pdf(dists[2], samples[:, sidx])
    p3 = pis[3] * pdf(dists[3], samples[:, sidx])
    
    if p1 > p2 && p1 > p3
        cur_z = "a"
    elseif p2 > p1 && p2 > p3
        cur_z = "b"
    else
        cur_z = "c"
    end
    push!(z, cur_z)
end
scatter_pred = Plots.Scatter(samples[1,:], samples[2,:], z, scatterClasses=sc)
push!(g, scatter_pred)

# 1d check
samples = zeros(1, num_samples)
gauss1 = Normal(-1.5, .6)
gauss2 = Normal(1.5, .6)
z = String[]
samp_w = Float64[]
for i in 1:size(samples, 2)
    if rand() > .5
        samples[:, i] = rand(gauss1)
        push!(z, "a")
        push!(samp_w, 1.)
    else
        samples[:, i] = rand(gauss2)
        push!(z, "b")
        push!(samp_w, 1.)
    end
end
scatter_orig = Plots.Scatter(samples[1,:], zeros(length(samples[1,:])), z, scatterClasses=sc)
push!(g, scatter_orig)

pis, mus, sigmas = fit_gmm(samples, samp_w = reshape(samp_w, 1, length(samp_w)))
println("pis: $(pis)")
println("mus: $(mus)")
println("sigmas: $(sigmas)")

N, K = size(samples, 2), length(pis)
dists = [MvNormal(mus[:,k], sigmas[:,:,k]) for k in 1:K]
z = String[]
for sidx in 1:N
    p1 = pis[1] * pdf(dists[1], samples[:, sidx])
    p2 = pis[2] * pdf(dists[2], samples[:, sidx])
    cur_z = p1 > p2 ? "a" : "b"
    push!(z, cur_z)
end
scatter_pred = Plots.Scatter(samples[1,:], zeros(length(samples[1,:])), z, scatterClasses=sc)
push!(g, scatter_pred)

g