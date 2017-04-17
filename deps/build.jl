# clone
urls = [
    "https://github.com/sisl/GridInterpolations.jl.git"
]

for url in urls
    try
        Pkg.clone(url)
    catch e
        println("Exception when cloning $(url): $(e)")  
    end
end