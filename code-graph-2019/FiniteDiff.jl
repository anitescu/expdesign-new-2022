include("./fgh_precise.jl")
using Pkg
using LinearAlgebra
using Random
using SparseArrays
using ForwardDiff
using Calculus
# This file is to makeu sure the gradient and hessian are correct, by comparing
# them with finite difference methods

function computeHess(M, nx_each, nt, nd, nr, gamma)
    H::Array{Float64, 2} = zeros(nd * nr, nd * nr)
    for i = 1 : (nd * nr)
        for j = 1 : i
            H[i, j] = tr(gamma \ M[j,:,:]' * F[((i - 1) * nt + 1): i * nt, :]' * (gamma \ M[i,:,:]' * M[j,:,:])) + tr(gamma \ M[i,:,:]' * F[((j - 1) * nt + 1): j * nt, :]' * (gamma \ M[j,:,:]' * M[i,:,:]))
            H[j, i] = H[i, j]
        end
    end
    Hess::Array{Float64, 2} = zeros(nd, nd)
    for i = 1 : nd
        for j = 1 : i
            Hess[i, j] = sum(sum(H[((i - 1) * nr + 1) : i * nr, ((j - 1) * nr + 1) : j * nr]))
            Hess[j, i] = Hess[i, j]
        end
    end
    return Hess
end

function obj(w, F, nx_each, nt, nd, nr, gamma)
    nx = nx_each * nx_each
    P = blockdiag()
    precision = inv(gamma)
    for i1::Int = 1 : nd
        for i2::Int = 1 : nr
            P = blockdiag(P, sparse(w[i1] * precision))
        end
    end
    eye = Matrix{Float64}(I, nx, nx)
    Mat = Symmetric(transpose(F) * P * F + eye)
    obj = tr(inv(Mat))
    return obj
end

# Define parameters in the problem
T = 1.0 # time range
nt = 5 # discetization parameter in time
gamma = Matrix{Float64}(I, nt, nt) # covariance matrix in time
cPDE1 = 0.1 # PDE parameter
cPDE2 = 0.1 # PDE parameter
muPDE = 1.0 # PDE parameter
p = 1 # truncated Fourier series
r = 0.2 # proportion of sensors vs candidate locations

nx_each = 5 # discretization parameter on each side
nx = nx_each * nx_each
nd = nx_each # discretization on the angle
nr = nx_each # discretization on the radius
println("nx_each = ", nx_each)
x_L = 1.0 * zeros(nd)
x_U = 1.0 * ones(nd)
beq = floor(r * nd)
g_L = [beq]
g_U = [beq]
F = mymodule.pto(cPDE1, cPDE2, muPDE, T, nx_each, nt, nd, nr, p)
Random.seed!(1234)
x = rand(nd)
M = mymodule.computeM(x, nx_each, nt, F, nd, nr, gamma)

prod_arr = mymodule.prod(F, x, nx_each, nt, nd, nr, gamma)
grad = mymodule.dertrace(prod_arr, gamma, nd, nr)
grad1 = mymodule.dertrace(M, gamma, nd, nr)
grad2 = mymodule.dertrace2(prod_arr, nd, nr, gamma)
hessian = computeHess(M, nx_each, nt, nd, nr, gamma)

f(x) = obj(x, F, nx_each, nt, nd, nr, gamma)
g1 = x -> ForwardDiff.gradient(f, x)
g2 = Calculus.gradient(f)
# h1 = x -> ForwardDiff.hessian(f, x)
# h2 = Calculus.hessian(f)
println("Forward diff gradient: ", g1(x))
println("Calculus gradient: ", g2(x))
println("Self computed gradient: ", grad)
println("Self computed gradient: ", grad1)
println("Self computed gradient2: ", grad2)
println("Forward diff hessian: ", h1(x))
println("Calculus hessian: ", h2(x))
println("Self computed hessian: ", hessian)
println("difference in gradient: ", norm(g1(x) - grad))
println("difference in hessian: ", norm(h1(x) - hessian))
