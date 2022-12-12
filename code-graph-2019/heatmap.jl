using Plots
pyplot()
n = 30
theta = LinRange(2 * pi / n, 2 * pi, n)
r = LinRange(1 / n, 1, n)
out = reshape(output, n, n)
# out = rand(15, 15) .- 1

# ax = PyPlot.axes(polar = "true")
# pcolormesh(theta, r, out)
# ax[:grid](true)
# savefig("heatmap.pdf")

heatmap(theta, r, out, proj = :polar)
# savefig("output.pdf")
