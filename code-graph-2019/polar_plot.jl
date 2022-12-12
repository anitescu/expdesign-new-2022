using Pkg
using DelimitedFiles
using Plots
x = readdlm("x.txt")
n = length(x)
m = Int(sum(x))
theta = LinRange(2 * pi / n, 2 * pi, n)
theta = x .* theta
angle = Array{Float64}(undef, 0)
for i = 1 : n
    if theta[i] != 0
        push!(angle, theta[i])
    end
end
println(angle)
r = 1.0 * ones(10)
Plots.plot(angle, r, proj = :polar, marker=:o, m=:red, bg=:black, lims=(0, 1.1))
Plots.title!("c = - 0.1, \\mu = 1.0, p = 3, n = 16")
savefig("polar2-2.pdf")
