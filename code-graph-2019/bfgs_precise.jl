include("./fg_bfgs.jl")
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
    coeff::Float64 = 1.0 # coefficient in the number of interpolation points
    # iteration in the number of interpolation points
    max_ite::Int = 2

    # iteration of mesh size
    m::Int = 4
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
    t_bfgs::Array{Float64, 2} = zeros(max_ite, m)
    # gap between upper bound and exact lower bound
    dval::Array{Float64, 2} = zeros(max_ite, m)
    # maximum KKT violation
    maxVio::Array{Float64, 2} = zeros(max_ite, m)

    # Solve the exact relaxed problem
    for i::Int = i_min : m
      nx_each = sqp.discParam(i, base) # discretization parameter on each side
      nx = nx_each * nx_each
      nd = nx_each # discretization on the angle
      nr = nx_each # discretization on the radius
      println("i = $(i), nx_each = ", nx_each)
      x_L = 1.0 * zeros(nd)
      x_U = 1.0 * ones(nd)
      beq = floor(r * nd)
      g_L = [beq]
      g_U = [beq]
      F = mymodule.pto(cPDE1, cPDE2, muPDE, T, nx_each, nt, nd, nr, p)
      eval_f(x) = mymodule.myfun1(x, F, nx_each, nt, nd, nr, gamma, vr)
      eval_grad_f(x, grad_f) = mymodule.myfun2(x, grad_f, F, nx_each, nt, nd, nr, gamma, vr)
      # eval_h(x, mode, rows, cols, obj_factor, lambda, values) = mymodule.myfun3(x, mode, rows, cols, obj_factor, lambda, values, F, nx_each, nt, nd, nr, gamma, vr)

      prob = createProblem(nd, x_L, x_U, 1, g_L, g_U, nd, Int64(nd * (nd + 1) / 2), eval_f, eval_g, eval_grad_f, eval_jac_g)
      addOption(prob, "hessian_approximation", "limited-memory")
      addOption(prob, "limited_memory_update_type", "bfgs")
      addOption(prob, "tol", 1e-6)
      addOption(prob, "max_iter", 100)
      prob.x = r * ones(nd)
      val, t[i] = @timed status = solveProblem(prob)
      x[i] = prob.x
      l_exact[i] = prob.obj_val
      # z[i] = mymodule.intapprox(nd, x[i])
      # u[i] =  mymodule.myfun1(z[i], F, nx_each, nt, nd, nr, gamma, vr)
    end

    ite::Int = 1
    while (ite <= max_ite)
        y, maxVio[ite, :], t_bfgs[ite, :], bytes, gctime = ipopt_bfgs(cPDE1, cPDE2, muPDE, T, p, r, m, coeff, base, i_min, nt, gamma, vr)
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
    return x, y, z, maxVio, t_bfgs, t, u, l_exact, l_approx, dval
end


# Call Ipopt with BFGS to solve the interpolation version
# --------------------------------------------------------------------------------------------------------
function ipopt_bfgs(cPDE1::Float64, cPDE2::Float64, muPDE::Float64, T::Float64,
    p::Int64, r::Float64, k::Int, coeff::Float64, base::Int64, i_min::Int,
    nt::Int64, gamma::Array{Float64, 2}, vr::Float64)
    # array to record sqp solutions
    w_arr = Array{Array{Float64}}(undef, k)
    for i::Int = 1 : (i_min - 1)
        w_arr[i] = [-1]
    end
    # maximum KKT violation
    maxVio::Array{Float64} = zeros(k)
    # computation time
    t::Array{Float64} = zeros(k)
    # total bytes allocated
    bytes::Array{Int64} = zeros(k)
    # garbage collection time
    gctime::Array{Float64} = zeros(k)
    # an object with various memory allocation counters
    memallocs = Array{Base.GC_Diff}(undef, k)

    for i::Int = i_min : k
        println("i = ", i)
        nx_each::Int = sqp.discParam(i, base)
        nx::Int64 = nx_each * nx_each
        nd::Int64 = nx_each
        nr::Int64 = nx_each
        println("mesh size of input domain = ", nx)
        println("number of variables = ", nd)
        n_each::Int = floor(sqrt(coeff * log(nx)))
        n::Int64 = n_each * n_each
        println("#interpolation points = ", n)

        # Call Ipopt with L-BFGS quasi-Newton methods for Hessian Approximation
        (Fns, Cnx) = fgh.InterpolationMatrix(n, nx, nd, nr, nt, cPDE1, cPDE2, muPDE, T, p)
        svdF = svd(Fns)
        Cn = sqp.SumCnx(Cnx, nd, nr, n)
        sqrt_gamma = sqrt(gamma)
        eval_f(x) = bfgs.EvalObj(svdF, x, Cnx, nd, nr, n, sqrt_gamma, vr, nt, nx)
        eval_grad_f(x, grad_f) = bfgs.EvalGradient(x, grad_f, Fns, Cnx, gamma, nx, nt, n, nd, nr, vr, svdF)

        x_L = 1.0 * zeros(nd)
        x_U = 1.0 * ones(nd)
        beq = floor(r * nd)
        g_L = [beq]
        g_U = [beq]
        prob = createProblem(nd, x_L, x_U, 1, g_L, g_U, nd, Int64(nd * (nd + 1) / 2), eval_f, eval_g, eval_grad_f, eval_jac_g)
        addOption(prob, "hessian_approximation", "limited-memory")
        addOption(prob, "limited_memory_update_type", "bfgs")
        addOption(prob, "tol", 1e-6)
        addOption(prob, "max_iter", 100)
        prob.x = r * ones(nd)
        val, t[i], bytes[i], gctime[i], memallocs[i] = @timed status = solveProblem(prob)
        w_arr[i] = prob.x

        # F::Array{Float64, 2} = mymodule.pto(cPDE1, cPDE2, muPDE, T, nx_each, nt, nd, nr, p)
        # maxVio[i] = mymodule.checkGap(w_arr[i], lambda, F, nx, nt, nd, nr, gamma, vr)
        # println("Max Violation = ", maxVio[i])
        # initial value, without calculating maxmimum violation
        maxVio[i] = -1
    end
    return w_arr, maxVio, t, bytes, gctime
end
