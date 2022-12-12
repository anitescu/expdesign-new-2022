using Gadfly
using DelimitedFiles
# w = readdlm("/Users/jingyu621/Documents/Code/Julia/Research/Mihai/2018_Computaion_GravitySurvey/2d_solver/data_and_graph/old/w.txt")
w = readdlm("/Users/jingyu621/Documents/Code/Julia/Research/Mihai/2018_Computaion_GravitySurvey/2d_solver/data_and_graph/new/w.txt")
i = 1
myplot = Gadfly.spy(reshape(w[i, 1 : 2500 * i * i], 50 * i, 50 * i))
Gadfly.draw(SVG("2d_trace_sensors.svg", 6inch, 6inch), myplot)

using PyPlot
t = readdlm("t.txt")
k = 10
s = 50 * (1 : k)
#plot(s, t, linestyle = "-", marker = "o")
PyPlot.semilogy(2 * log2(s), t, linestyle = "-", marker = "o", label = "Solve Time")
PyPlot.semilogy(2 * log2(s), 2 * s .* s.* log(s), linestyle = "-", marker = "o", label = "nlog(n)")
#xticks(1 : k)
xlabel("log2(n)")
ylabel("time in seconds")
legend()
grid("on")

bytes = readdlm("data_and_graph/new/bytes.txt")
PyPlot.semilogy(2 * log2(s), bytes, linestyle = "-", marker = "o", label = "real allocation")
PyPlot.semilogy(2 * log2(s), 1e8 * s .* s .* log(s) .* log(s), linestyle = "-", marker = "o", label = "c*nlog(n)")
xlabel("log2(n)")
ylabel("total bytes allocated")
legend()
grid("on")

# plot difference in the objective to show solution from sqp is good enough
clf()
dval = readdlm("dval.txt")
# dval[4, 6] = dval[4, 5]
labels = ["SQP with c=1", "SQP with c=2", "SQP with c=4", "SQP with c=8"]
for i = 1 : 4
    PyPlot.semilogy(5 .+ 5 * (1 : 7), abs.(dval[i, 2:end]), marker = "o", label = labels[i])
end
xlabel("discretization parameter on each side")
ylabel("error from SQP with interpolation")
# ylim([1e-5, 2e1])
# yticks(0.1 : 0.1 : 1.0)
# ylabel("max KKT violation")
legend()
# grid("on")
savefig("dval.pdf")

# plot KKT max violation to show solution from sqp is good enough
clf()
maxVio = readdlm("data2/maxVio.txt")
labels = ["SQP with c=1", "SQP with c=2", "SQP with c=4", "SQP with c=8"]
for i = 1 : 4
    PyPlot.semilogy(5 .+ 5 * (1 : 7), maxVio[i, 2:end], marker = "o", label = labels[i])
end
xlabel("discretization parameter on each side")
# ylabel("error from SQP with interpolation")
# yticks(0.1 : 0.1 : 1.0)
ylabel("max KKT violation")
legend()
# grid("on")
savefig("maxVio.pdf")

# plot computation time
clf()
t = readdlm("t.txt")
t_sqp = readdlm("t_sqp.txt")
labels = ["SQP with c=1", "SQP with c=2", "SQP with c=4", "SQP with c=8"]
for i = 1 : 4
    PyPlot.semilogy(5 .+ 5 * (1 : 7), t_sqp[i, 2:end], marker = "o", label = labels[i])
end
PyPlot.semilogy(5 .+ 5 * (1 : 7), t[2:end], marker = "o", label = "Exact method with Ipopt solver")
xlabel("discretization parameter on each side")
ylabel("computatin time (in seconds)")
legend()
# grid("on")
savefig("t.pdf")

# plot integrality gap (full) to show solution from sqp is good enough
clf()
u = readdlm("u_approx_full.txt")
l = readdlm("l_exact.txt")
labels = ["SQP with c=1", "SQP with c=2", "SQP with c=4", "SQP with c=8"]
for i = 1 : 4
    PyPlot.semilogy(5 .+ 5 * (1 : 7), abs.(u[i, 2:end] - l[2:end]), marker = "o", label = labels[i])
end
xlabel("discretization parameter on each side")
ylabel("integrality gap for SQP solution (full)")
# ylim([1e-5, 2e1])
# yticks(0.1 : 0.1 : 1.0)
# ylabel("max KKT violation")
legend()
grid("on")
savefig("intgap_full2.pdf")

# plot integrality gap (full) to show solution from sqp is good enough
clf()
u = readdlm("u_approx.txt")
l = readdlm("l_approx.txt")
dval = u - l
labels = ["SQP with c=1", "SQP with c=2", "SQP with c=4", "SQP with c=8"]
PyPlot.semilogy(5 .+ 5 * (1 : 11), dval[1, 2:end], marker = "o", label = labels[1])
PyPlot.semilogy(5 .+ 5 * [1 : 2; 4 : 11], [dval[2,2:3];dval[2,5:12]], marker = "o", label = labels[2])
PyPlot.semilogy(5 .+ 5 * (1 : 11), dval[3, 2:end], marker = "o", label = labels[3])
PyPlot.semilogy(5 .+ 5 * (1 : 11), dval[4, 2:end], marker = "o", label = labels[4])
xlabel("discretization parameter on each side")
ylabel("integrality gap for SQP solution (low rank)")
# ylim([1e-5, 2e1])
# yticks(0.1 : 0.1 : 1.0)
# ylabel("max KKT violation")
legend()
grid("on")
savefig("intgap_lowrank.pdf")

# plot gap in the objective to show sqp solution is good enough
clf()
u = readdlm("l_approx_full.txt")
l = readdlm("l_exact.txt")
labels = ["SQP with c=1", "SQP with c=2", "SQP with c=4", "SQP with c=8"]
for i = 1 : 4
    PyPlot.semilogy(5 .+ 5 * (1 : 7), abs.(u[i, 2:8] - l[2:end]), marker = "o", label = labels[i])
end
xlabel("discretization parameter on each side")
ylabel("integrality gap for SQP solution (full)")
# ylim([1e-5, 2e1])
# yticks(0.1 : 0.1 : 1.0)
legend()
grid("on")
savefig("gap_relax.pdf")
