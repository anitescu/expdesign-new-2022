include("./fgh_precise.jl")
include("./fgh.jl")
include("./ip_main.jl")

using LinearAlgebra
using Ipopt

# evaluate the linear constraint
function eval_g(x, g)
    g[1] = sum(x)
end

# evaluate the gradient of linear constraint
function eval_jac_g(x, mode, rows, cols, values)
  n = length(x)
  if mode == :Structure
    # Constraint (row) 1
    for i = 1 : n
      rows[i] = 1
      cols[i] = i
    end
  else
    # Constraint (row) 1
    for i = 1 : n
      values[i] = 1
    end
  end
end

function main()
    T::Float64 = 1.0 # time range
    nt::Int64 = 5 # discetization parameter in time
    gamma::Array{Float64, 2} = Matrix{Float64}(I, nt, nt) # covariance matrix in time
    cPDE1::Float64 = 0.1 # PDE parameter
    cPDE2::Float64 = 0.0 # PDE parameter
    muPDE::Float64 = 1.0 # PDE parameter
    p::Int64 = 3 # truncated Fourier series
    r::Float64 = 0.2 # proportion of sensors vs candidate locations
    vr::Float64 = 0.01 # noise ratio = measurement noise / prior noise
    coeff::Float64 = 8.0 # coefficient in the number of interpolation points
    # iteration in the number of interpolation points
    max_ite::Int = 1

    # iteration of mesh size
    m::Int = 6
    # parameter in mesh size
    base::Int = 5
    # y = solution to relaxed problem with interpolation method
    y = Array{Array{Float64}}(undef, max_ite, m)
    # SUR based on approx relaxed solution
    z = Array{Array{Float64}}(undef, max_ite, m)
    # computatin time for SQP
    t_sqp::Array{Float64, 2} = zeros(max_ite, m)
    # (low-rank) objective value of SQP solution
    l_approx::Array{Float64} = zeros(max_ite, m)
    # (full) objective value of SQP solution
    l_approx_full::Array{Float64} = zeros(max_ite, m)
    # (low rank) objective value of SUR SQP solution
    u_approx::Array{Float64} = zeros(max_ite, m)
    # (full) objective value of SUR SQP solution
    u_approx_full::Array{Float64} = zeros(max_ite, m)
    # number of iterations
    iter_num::Array{Float64} = zeros(max_ite, m)
    # the index starts at
    i_min::Int = 6

    ite::Int = 1
    while (ite <= max_ite)
        ytmp, maxVio, t_sqp[ite, :], bytes, gctime, iter_num[ite, :] = sqp.main(cPDE1, cPDE2, muPDE, T, p, r, m, coeff, base, i_min, nt, gamma, vr)
        # println(ytmp)
        y[ite, :] = ytmp
        nd = sqp.discParam(m, base)
        for i = i_min : m
            println("evaluate obj, i = ", i)
            nx_each::Int64 = sqp.discParam(i, base)
            nx::Int64 = nx_each * nx_each
            nd::Int64 = nx_each
            nr::Int64 = nx_each
            n_each::Int = floor(sqrt(coeff * log(nx)))
            n::Int64 = n_each * n_each
            z[ite, i] = mymodule.intapprox(nd, y[ite, i])
            # F = mymodule.pto(cPDE1, cPDE2, muPDE, T, nx_each, nt, nd, nr, p)
            # # u_approx_full[ite, i] = mymodule.myfun1(z[ite, i], F, nx_each, nt, nd, nr, gamma, vr)
            # l_approx_full[ite, i] = mymodule.myfun1(y[ite, i], F, nx_each, nt, nd, nr, gamma, vr)
            # (Fns, Cnx) = fgh.InterpolationMatrix(n, nx, nd, nr, nt, cPDE1, cPDE2, muPDE, T, p)
            # svdF = svd(Fns)
            # sqrt_gamma = sqrt(gamma)
            # u_approx[ite, i] = sqp.eval_obj(svdF, z[ite, i], Cnx, nd, nr, n, sqrt_gamma, vr, nt, nx)
            # l_approx[ite, i] = sqp.eval_obj(svdF, y[ite, i], Cnx, nd, nr, n, sqrt_gamma, vr, nt, nx)
        end
        ite += 1
        coeff *= 2
    end
    return y, z, t_sqp, l_approx, l_approx_full, u_approx, u_approx_full, iter_num
end
