# Implement SQP framework

module sqp

include("./fgh.jl")
include("./interior_point.jl")
include("./interior_point_LP.jl")
include("./fgh_precise.jl")

using LinearAlgebra
using SparseArrays
using Ipopt
using JuMP

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

# evaluate the gradient of the objective, which is a constant vector
function eval_grad_obj(x, grad_f, c)
    grad_f[:] = c
    # println("grad_f:", grad_f)
end

# evaluate the linear objective value
function eval_obj_linear(c, x)
    # println("obj:", dot(c, x))
    return dot(c, x)
end

# Compute KKT violation for the approximate F matrix problem is satisfied
# --------------------------------------------------------------------------------------------------------
function Check_KKT(w::Array{Float64}, c::Array{Float64}, u::Array{Float64},
    l::Array{Float64}, n0::Float64, lambda::Array{Float64}, m::Int64, nx::Int64)
    rho::Float64 = lambda[end]
    eps::Float64 = abs(rho * (sum(w) - n0))
    eps1::Float64 = 0.0
    eps2::Float64 = 0.0
    eps3::Float64 = 0.0
    for i::Int = 1 : nx
        eps1 = maximum([eps1, abs(c[i] + rho - lambda[i] + lambda[i + nx])])
        eps2 = maximum([eps2, abs(lambda[i] * (- w[i]))])
        eps3 = maximum([eps3, abs(lambda[i + nx] * (w[i] - 1))])
    end
    return maximum([eps, eps1, eps2, eps3])
end

# Aggregate interpolation coefficient matrix because points along the same beam
# directin share the same wight
# --------------------------------------------------------------------------------------------------------
function SumCnx(Cnx::Array{Float64, 2}, nd::Int64, nr::Int64, n::Int64)
    Cn::Array{Float64, 2} = zeros(n, nd)
    for i::Int = 1 : nd
        for j::Int = 1 : nr
            Cn[:, i] += Cnx[:, (i - 1) * nr + j]
        end
    end
    return Cn
end

# Compute trace inverse objective value with svd decomposition
# --------------------------------------------------------------------------------------------------------
function eval_obj(svdF::SVD, w::Array{Float64}, Cnx::Array{Float64, 2},
    nd::Int64, nr::Int64, n::Int64, sqrt_gamma::Array{Float64, 2}, vr::Float64,
    nt::Int64, nx::Int64)
    # default gamma = identity
    # sqrtW = kron(Diagonal(sqrt.(abs.(w))), Matrix{Float64}(I, nr, nr), sqrt_gamma)
    # wCnx = kron(Cnx, Matrix{Float64}(I, nt, nt)) * sqrtW
    sqrt_w = sqrt.(abs.(w))
    Cnxw = similar(Cnx)
    for i::Int = 1 : nd
        Cnxw[:, ((i - 1) * nr + 1) : i * nr] = Cnx[:, ((i - 1) * nr + 1) : i * nr] * sqrt_w[i]
    end
    wCnx = kron(Cnxw, sqrt_gamma)

    svdWC = svd(wCnx)
    eigenM = Diagonal(svdF.S) * svdF.Vt * svdWC.U * Diagonal(svdWC.S .^ 2) * svdWC.U' * svdF.Vt' * Diagonal(svdF.S)
    eigenval = real(eigvals(eigenM))
    eigenv = zeros(nx)
    eigenv[1 : size(eigenval)[1]] = eigenval
    # double check the objective is correct
    # obj1 = sum(1 ./ (vr .+ eigenv))
    # F::Array{Float64, 2} = transpose(svdF.U * Diagonal(svdF.S) * svdF.Vt * kron(Cnx, Matrix{Float64}(I, nt, nt)))
    # nx_each::Int = sqrt(size(F)[2])
    # obj2 = mymodule.myfun1(w, F, nx_each, nt, nd, nr, gamma, vr)
    # println("obj1:", obj1)
    # println("obj2:", obj2)
    return sum(1 ./ (vr .+ eigenv))
end

# Implement line search to select step size
# --------------------------------------------------------------------------------------------------------
function line_search(fval0::Float64, c::Array{Float64,1}, dw::Array{Float64,1},
    svdF::SVD, w::Array{Float64,1}, Cnx::Array{Float64, 2}, nd::Int64, nr::Int64,
    n::Int64, sqrt_gamma::Array{Float64, 2}, vr::Float64, nt::Int64,
    dl::Array{Float64}, nx::Int64)
    alpha::Float64 = 1.0
    println("sum(dw): ", sum(dw))
    wnew = w + alpha * dw
    fval = eval_obj(svdF, wnew, Cnx, nd, nr, n, sqrt_gamma, vr, nt, nx)
    while fval > fval0 + 1e-3 * alpha * c' * dw
        # println("alpha:", alpha)
        alpha *= 0.5
        wnew = w + alpha * dw
        fval = eval_obj(svdF, wnew, Cnx, nd, nr, n, sqrt_gamma, vr, nt, nx)
    end
    obj_diff = fval0 - fval
    println("alpha: ", alpha)
    # println("decrease in obj: ", onj_diff)
    return fval, wnew, alpha * dw, alpha * dl, obj_diff
end

# --------------------------------------------------------------------------------------------------------
# Implement SQP algorithm
function main_each_step(Fns::Array{Float64}, Cnx::Array{Float64}, r::Float64,
    nx::Int64, n::Int64, nt::Int64, nd::Int64, nr::Int64,
    gamma::Array{Float64, 2}, vr::Float64)
    # (Fns, Cnx) = fgh.InterpolationMatrix(n, nx, nd, nr, nt, cPDE1, cPDE2, muPDE, T, p)
    svdF = svd(Fns)
    Cn = SumCnx(Cnx, nd, nr, n)
    sqrt_gamma = sqrt(gamma)

    m::Int = 2 * nd + 1
    ds::Float64 = (2 / Int(sqrt(nx))) ^ 2
    r = floor(r * nd) / nd

    w::Array{Float64} = r * ones(nd)
    u::Array{Float64} = ones(nd) - w
    l::Array{Float64} = zeros(nd) - w
    n0::Float64 = sum(w)
    c0::Float64 = n0 - sum(w)

    k::Int = 1
    # println("time for running CG: ")
    (M1, preH) = fgh.RunCG(w, Fns, Cnx, gamma, nx, nt, n, nd, nr, vr, svdF)
    # println("time for approx gradient: ")
    c = fgh.GradientApprox(w, M1, Cnx, nd, nr)
    # println("time for IP: ")
    lambda = zeros(m) # initialize Lagrangian multipliers
    (dw, y, new_lambda) = interior_point.InteriorPoint(Cn, preH, c, u, l, c0, m, nd, ds)
    dl = new_lambda - lambda
    # for i = 1 : nd
    #     if w[i] + dw[i] < 0
    #         println("ERROR! w < 0, i = ", i, ", w[i] = ", w[i], ", dw[i] = ", dw[i], ", new w[i] = ", w[i] + dw[i])
    #     end
    #     if w[i] + dw[i] > 1
    #         println("ERROR! w > 1, i = ", i, ", w[i] = ", w[i], ", dw[i] = ", dw[i], ", new w[i] = ", w[i] + dw[i])
    #     end
    # end

    # do line search instead of newton's step w = w + dw
    fval0 = eval_obj(svdF, w, Cnx, nd, nr, n, gamma, vr, nt, nx)
    fval0, w, dw, dl, obj_diff = line_search(fval0, c, dw, svdF, w, Cnx, nd, nr, n, sqrt_gamma, vr, nt, dl, nx)
    lambda += dl

    u = ones(nd) - w
    l = zeros(nd) - w
    c0 = n0 - sum(w)
    # println("time for running CG: ")
    (M1, preH) = fgh.RunCG(w, Fns, Cnx, gamma, nx, nt, n, nd, nr, vr, svdF)
    # println("time for approx gradient: ")
    c = fgh.GradientApprox(w, M1, Cnx, nd, nr)
    # println("time to check KKT:")
    # opt_violation = Check_KKT(w, c, u, l, n0, lambda, m, nd)

    println("dw:", norm(dw, Inf))
    println("obj_diff:", obj_diff)
    # println("KKT optimality violation: ", opt_violation)

    # x1::Array{Float64} = dw
    # x2::Array{Float64} = dw
    while (obj_diff > 1e-3)
        # println("time for IP: ")
        (dw, y, new_lambda) = interior_point.InteriorPoint(Cn, preH, c, u, l, c0, m, nd, ds)
        dl = new_lambda - lambda
        # for i = 1 : nd
        #     if w[i] + dw[i] < 0
        #         println("ERROR! w < 0, i = ", i, ", w[i] = ", w[i], ", dw[i] = ", dw[i], "new w[i] = ", w[i] + dw[i])
        #     end
        #     if w[i] + dw[i] > 1
        #         println("ERROR! w > 1, i = ", i, ", w[i] = ", w[i], ", dw[i] = ", dw[i], "new w[i] = ", w[i] + dw[i])
        #     end
        # end

        (dw, y, new_lambda) = interior_point_LP.InteriorPoint(c, u, l, c0, m, nd, ds)
        dl = new_lambda - lambda

        println("New iterate, k = ", k)
        k += 1
        # do line search instead of taking newton's step w = w + dw
        fval0, w, dw, dl, obj_diff = line_search(fval0, c, dw, svdF, w, Cnx, nd, nr, n, sqrt_gamma, vr, nt, dl, nx)
        lambda += dl
        println("dw:", norm(dw, Inf))
        println("obj_diff:", obj_diff)

        # x2 = x1
        # x1 = dw
        #
        # println("norm(x1 + x2)=", norm(x1 + x2))
        # if (norm(x1 + x2, Inf) < 1e-5)
        #     println("Observe moving back and forth!")
        #     break
        # end

        u = ones(nd) - w
        l = zeros(nd) - w
        c0 = n0 - sum(w)
        # println("time for running CG: ")
        (M1, preH) = fgh.RunCG(w, Fns, Cnx, gamma, nx, nt, n, nd, nr, vr, svdF)
        # println("time for approx gradient: ")
        c = fgh.GradientApprox(w, M1, Cnx, nd, nr)
        # println("time to check KKT:")
        # opt_violation = Check_KKT(w, c, u, l, n0, lambda, m, nd)
        # println("KKT optimality violation: ", opt_violation)

# --------------------------------------------------------------------------------------------------------
# add steepest descent step
        # if obj_diff < 1e-2
        #     # # Call Ipopt to solve LP
        #     # eval_f(x) = eval_obj_linear(c, x)
        #     # eval_grad_f(x, grad_f) = eval_grad_obj(x, grad_f, c)
        #     #
        #     # prob = createProblem(nd, l, u, 1, [- Inf], [c0], nd, 0, eval_f, eval_g, eval_grad_f, eval_jac_g)
        #     # addOption(prob, "tol", 1e-6)
        #     # addOption(prob, "max_iter", 100)
        #     # addOption(prob, "hessian_approximation", "limited-memory")
        #     # prob.x = zeros(nd)
        #     # status = solveProblem(prob)
        #     # dw = prob.x
        #
        #     # # Call JuMP wrapper to solve LP
        #     # model = Model(with_optimizer(Ipopt.Optimizer))
        #     # @variable(model, x[i = 1 : nd])
        #     # for i = 1 : nd
        #     #     @constraint(model, l[i] <= x[i] <= u[i])
        #     # end
        #     # @constraint(model, sum(x) == c0)
        #     # @objective(model, Min, dot(c, x))
        #     # JuMP.optimize!(model)
        #     # dw = JuMP.value.(x)
        #
        #     (dw, y, new_lambda) = interior_point_LP.InteriorPoint(c, u, l, c0, m, nd, ds)
        #
        #     println("Call steepest descent:")
        #     println("dw:", norm(dw, Inf))
        #     fval0, w, dw, dltmp, obj_diff = line_search(fval0, c, dw, svdF, w, Cnx, nd, nr, n, sqrt_gamma, vr, nt, dl, nx)
        #     println("obj_diff:", obj_diff)
        #
        #     u = ones(nd) - w
        #     l = zeros(nd) - w
        #     c0 = n0 - sum(w)
        #     # println("time for running CG: ")
        #     (M1, preH) = fgh.RunCG(w, Fns, Cnx, gamma, nx, nt, n, nd, nr, vr, svdF)
        #     # println("time for approx gradient: ")
        #     c = fgh.GradientApprox(w, M1, Cnx, nd, nr)
        # end
# --------------------------------------------------------------------------------------------------------
    end

    return w, lambda, k
end

# --------------------------------------------------------------------------------------------------------
# number of discretization points on each side
function discParam(i::Int, base::Int)
    return base * i
end

# --------------------------------------------------------------------------------------------------------
# main function
function main(cPDE1::Float64, cPDE2::Float64, muPDE::Float64, T::Float64, p::Int64,
    r::Float64, k::Int, coeff::Float64, base::Int64, i_min::Int, nt::Int64,
    gamma::Array{Float64, 2}, vr::Float64)
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
    # number of iterations
    iter_num = zeros(k)
    for i::Int = i_min : k
        println("i = ", i)
        nx_each::Int = discParam(i, base)
        nx::Int64 = nx_each * nx_each
        nd::Int64 = nx_each
        nr::Int64 = nx_each
        println("mesh size of input domain = ", nx)
        println("number of variables = ", nd)
        n_each::Int = floor(sqrt(coeff * log(nx)))
        n::Int64 = n_each * n_each
        println("#interpolation points = ", n)
        (Fns, Cnx) = fgh.InterpolationMatrix(n, nx, nd, nr, nt, cPDE1, cPDE2, muPDE, T, p)
        res, t[i], bytes[i], gctime[i], memallocs[i] = @timed main_each_step(Fns, Cnx, r, nx, n, nt, nd, nr, gamma, vr)

        w_arr[i] = res[1]
        iter_num[i] = res[3]
        # lambda = res[2]
        # F::Array{Float64, 2} = mymodule.pto(cPDE1, cPDE2, muPDE, T, nx_each, nt, nd, nr, p)
        # maxVio[i] = mymodule.checkGap(w_arr[i], lambda, F, nx, nt, nd, nr, gamma, vr)
        # println("Max Violation = ", maxVio[i])
        # initial value, without calculating maxmimum violation
        maxVio[i] = -1
    end
    return w_arr, maxVio, t, bytes, gctime, iter_num
end
# Profile.clear()
# @profile w_arr, t, bytes, gctime = main();
# Profile.print(format =:flat)
end
