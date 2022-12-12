include("./fgh_precise.jl")
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
    coeff::Float64 = 1.0 # coefficient in the number of interpolation points
    # iteration in the number of interpolation points
    max_ite::Int = 4

    # iteration of mesh size
    m::Int = 8
    # parameter in mesh size
    base::Int = 5
    # x = solution to exact relaxed problem
    x = Array{Array{Float64}}(undef, m)
    # y = solution to relaxed problem with interpolation method
    y = Array{Array{Float64}}(undef, m)
    # SUR based on exact relaxed solution
    z = Array{Array{Float64}}(undef, m)
    # iteration start index
    i_min::Int = 2
    # fill in undef values for indices < i_min
    for i::Int = 1 : (i_min - 1)
        x[i] = [-1]
        y[i] = [-1]
        z[i] = [-1]
    end
    # lower bound from excat relaxed solution
    l_exact::Array{Float64} = zeros(m)
    # lower bound from approx relaxed solution (SQP)
    l_approx::Array{Float64} = zeros(max_ite, m)
    # upper bound from SUR
    u::Array{Float64} = zeros(m)
    # computation time for exact method
    t::Array{Float64} = zeros(m)
    # computation time for SQP method
    t_sqp::Array{Float64, 2} = zeros(max_ite, m)
    # gap between upper bound and exact lower bound
    dval::Array{Float64, 2} = zeros(max_ite, m)
    # maximum KKT violation
    maxVio::Array{Float64, 2} = zeros(max_ite, m)

    # # Solve the exact relaxed problem
    # for i::Int = i_min : m
    #   nx_each = sqp.discParam(i, base) # discretization parameter on each side
    #   nx = nx_each * nx_each
    #   nd = nx_each # discretization on the angle
    #   nr = nx_each # discretization on the radius
    #   println("i = $(i), nx_each = ", nx_each)
    #   x_L = 1.0 * zeros(nd)
    #   x_U = 1.0 * ones(nd)
    #   beq = floor(r * nd)
    #   g_L = [beq]
    #   g_U = [beq]
    #   F = mymodule.pto(cPDE1, cPDE2, muPDE, T, nx_each, nt, nd, nr, p)
    #   eval_f(x) = mymodule.myfun1(x, F, nx_each, nt, nd, nr, gamma, vr)
    #   eval_grad_f(x, grad_f) = mymodule.myfun2(x, grad_f, F, nx_each, nt, nd, nr, gamma, vr)
    #   eval_h(x, mode, rows, cols, obj_factor, lambda, values) = mymodule.myfun3(x, mode, rows, cols, obj_factor, lambda, values, F, nx_each, nt, nd, nr, gamma, vr)
    #
    #   prob = createProblem(nd, x_L, x_U, 1, g_L, g_U, nd, Int64(nd * (nd + 1) / 2), eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    #   addOption(prob, "tol", 1e-6)
    #   addOption(prob, "max_iter", 100)
    #   # addOption(prob, "hessian_approximation", "limited-memory")
    #   prob.x = r * ones(nd)
    #   val, t[i] = @timed status = solveProblem(prob)
    #   x[i] = prob.x
    #   l_exact[i] = prob.obj_val
    #   z[i] = mymodule.intapprox(nd, x[i])
    #   u[i] =  mymodule.myfun1(z[i], F, nx_each, nt, nd, nr, gamma, vr)
    # end

    ite::Int = 1
    while (ite <= max_ite)
        y, maxVio[ite, :], t_sqp[ite, :], bytes, gctime = sqp.main(cPDE1, cPDE2, muPDE, T, p, r, m, coeff, base, i_min, nt, gamma, vr)
        # compute optimality gap
        for i::Int = i_min : m
            nx_each = sqp.discParam(i, base)
            nd = nx_each
            nr = nx_each
            F = mymodule.pto(cPDE1, cPDE2, muPDE, T, nx_each, nt, nd, nr, p)
            l_approx[ite, i] = mymodule.myfun1(y[i], F, nx_each, nt, nd, nr, gamma, vr)
            dval[ite, i] = l_approx[ite, i] - l_exact[i]
        end
        ite += 1
        coeff *= 2
    end
    return x, y, z, maxVio, t_sqp, t, u, l_exact, l_approx, dval
end
