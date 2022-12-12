module interior_point
include("./fgh.jl")
include("./interior_point_LP.jl")
using LinearAlgebra

# --------------------------------------------------------------------------------------------------------
# Compute A^T * x
function ATx(x::Array{Float64}, nx::Int64, m::Int64)
    ATx::Array{Float64} = zeros(nx)
    for i::Int = 1 : nx
        ATx[i] = x[i] - x[nx + i] - x[m]
    end
    return ATx
end

# --------------------------------------------------------------------------------------------------------
# Compute part of RHS in (16.62)
function FindVec(rp::Array{Float64}, y::Array{Float64}, lambda::Array{Float64},
    sigma::Float64, mu::Float64, m::Int64)
    vec::Array{Float64} = zeros(m)
    sm::Float64 = sigma * mu
    # println("last elements in y:", y[m])
    # println("last elements in lambda:", lambda[m])
    for i::Int = 1 : m
        vec[i] = - lambda[i] * rp[i] / y[i] - lambda[i] + sm / y[i]
    end
    return vec
end

# --------------------------------------------------------------------------------------------------------
# Solve for Inv(D)*x
function FindInvD(x::Array{Float64}, y::Array{Float64}, lambda::Array{Float64}, nx::Int64)
    InvDx::Array{Float64} = zeros(nx)
    for i::Int = 1 : nx
        InvDx[i] = x[i] / (lambda[i] / y[i] + lambda[i + nx] / y[i + nx])
    end
    return InvDx
end

# --------------------------------------------------------------------------------------------------------
# Matrix vector product in the innerCG step
function innerCG_MatVecProd(x::Array{Float64}, d::Array{Float64}, nx::Int64,
    preH::Array{Float64, 2}, Cn::Array{Float64, 2})
    xtmp::Array{Float64} = preH \ x
    x = Cn' * x
    x = x ./ d
    x = Cn * x
    return x + xtmp
end


# --------------------------------------------------------------------------------------------------------
# Conjugate gradient to compute inv(C'*inv(D)*C+inv(preH)) * vec
function innerCG(vec::Array{Float64}, d::Array{Float64},
    preH::Array{Float64, 2}, nx::Int64, Cn::Array{Float64, 2}, n::Int64)
    # println("CG!")
    x::Array{Float64} = zeros(n)
    r::Array{Float64} = - vec
    p = - r
    while (norm(r) > 1e-8)
        #println(norm(r))
        q::Array{Float64} = innerCG_MatVecProd(p, d, nx, preH, Cn)
        c::Float64 = dot(r, r)
        a::Float64 = c / dot(p, q)
        x = x + a * p
        r = r + a * q
        b::Float64 = dot(r, r) / c
        p = - r + b * p
    end

    return x
end

# --------------------------------------------------------------------------------------------------------
# Solve for Inv(B)*x
function FindInvB(x::Array{Float64}, Cn::Array{Float64, 2}, preH::Array{Float64, 2},
    y::Array{Float64}, lambda::Array{Float64}, nx::Int64, n::Int64, CDC::Array{Float64, 2})
    InvDx = FindInvD(x, y, lambda, nx)
    xtmp::Array{Float64} = InvDx
    xtmp = Cn * xtmp
    xtmp = CDC \ xtmp
    # implement another CG to compute (inv(CDC + inv(preH)))
    # dtmp = lambda ./ y
    # d = dtmp[1 : nx] + dtmp[(nx + 1) : (nx + nx)]
    # xtmp = innerCG(xtmp, d, preH, nx, Cn, n)
    # diff = norm(xtmp - xtmp1)
    # if diff > 1e-8
    #     prinln("Large difference!", diff)
    # end
    # @everywhere Cn, lambda, y, nx
    # CDC = @parallel (+) for i = 1 : nx
    # Cn[:, i] * Cn[:, i]' / (lambda[i] / y[i] + lambda[i + nx] / y[i + nx])
    xtmp = Cn' * xtmp
    xtmp = FindInvD(xtmp, y, lambda, nx)
    return InvDx - xtmp
end

# --------------------------------------------------------------------------------------------------------
# Solve for dx
function SolveDx(Cn::Array{Float64, 2}, preH::Array{Float64, 2}, y::Array{Float64},
    lambda::Array{Float64}, rhs::Array{Float64}, m::Int64, nx::Int64, n::Int64)
    mu_const = lambda[m] / y[m]
    v0::Array{Float64} = ones(nx)
    CDC::Array{Float64, 2} = zeros(n, n)
    for i::Int = 1 : nx
        CDC = CDC + Cn[:, i] * Cn[:, i]' / (lambda[i] / y[i] + lambda[i + nx] / y[i + nx])
    end
    CDC += inv(preH)
    v1::Array{Float64} = FindInvB(v0, Cn, preH, y, lambda, nx, n, CDC)
    v2::Array{Float64} = FindInvB(rhs, Cn, preH, y, lambda, nx, n, CDC)
    return v2 - v1 * dot(v0, v2) / (dot(v0, v1) + 1 / mu_const)
end

# --------------------------------------------------------------------------------------------------------
# Solve a linear system (16.58) for dx, dy and dl
function SolveLinearSysm(Cn::Array{Float64, 2}, preH::Array{Float64, 2},
    c::Array{Float64}, b::Array{Float64}, x::Array{Float64}, y::Array{Float64},
    lambda::Array{Float64}, sigma::Float64, mu::Float64, m::Int64, nx::Int64, n::Int64)
    Gx::Array{Float64} = Cn * x
    Gx = preH * Gx
    Gx = Cn' * Gx
    rd::Array{Float64} = Gx - ATx(lambda, nx, m) + c
    Ax::Array{Float64} = vcat(x, - x, - sum(x))
    rp::Array{Float64} = Ax - y - b
    vec::Array{Float64} = FindVec(rp, y, lambda, sigma, mu, m)
    rhs::Array{Float64} = - rd + ATx(vec, nx, m)
    dx::Array{Float64} = SolveDx(Cn, preH, y, lambda, rhs, m, nx, n)
    dy::Array{Float64} = vcat(dx, - dx, - sum(dx)) + rp
    # v0::Array{Float64} = ones(m)
    v1::Array{Float64} = lambda./ y
    # dl::Array{Float64} = - lambda + sigma * mu * (v0./ y) - v1.* dy
    dl::Array{Float64} = - lambda + sigma * mu ./ y - v1.* dy

    Gdx::Array{Float64} = Cn * dx
    Gdx = preH * Gdx
    Gdx = Cn' * Gdx
    ATlamb = ATx(dl, nx, m)
    test = Gdx - ATlamb + rd
    # if norm(test) > 1e-3
    #     println("check norm = ", norm(test))
    # end
    return dx, dy, dl
end

# --------------------------------------------------------------------------------------------------------
# Find maximum alpha in (0, 1]
function SolveAlpha(dx::Array{Float64}, rhs::Array{Float64}, m::Int64)
    alpha::Float64 = 1.0
    for i::Int = 1 : m
        if dx[i] < 0
            alpha = min(alpha, rhs[i] / dx[i])
        end
    end
    if alpha < 0
        println("Error! Negative alpha.")
    end
    return alpha
end

# --------------------------------------------------------------------------------------------------------
# Find Initial Point in Interior Point Method
function InitialPoint(Cn::Array{Float64, 2}, preH::Array{Float64, 2},
    c::Array{Float64}, b::Array{Float64}, m::Int64, nx::Int64, n::Int64)
    x0::Array{Float64} = zeros(nx)
    y0::Array{Float64} = 0.001 * ones(m)
    lambda0::Array{Float64} = 0.001 * ones(m)
    mu = dot(y0, lambda0) / m
    #println("In initial IP, time to solve a linear system:")
    (dx_aff, dy_aff, dl_aff) = SolveLinearSysm(Cn, preH, c, b, x0, y0, lambda0, 0.0, mu, m, nx, n)
    for i::Int = 1 : m
        y0[i] = max(1, abs(y0[i] + dy_aff[i]))
        lambda0[i] = max(1, abs(lambda0[i] + dl_aff[i]))
    end
    return x0, y0, lambda0
end

# --------------------------------------------------------------------------------------------------------
# Implementation of Interior Point Method to Solve a QP (Algorithm 16.4 in Nocedal & Wright)
function InteriorPoint(Cn::Array{Float64, 2}, preH::Array{Float64, 2},
    c::Array{Float64}, u::Array{Float64}, l::Array{Float64}, c0::Float64,
    m::Int64, nx::Int64, ds::Float64)
    # nx is the dimension of the variable, not necessarily the mesh size
    # n is the dimension of preH, c*log(nx)
    # m is the number of constraints, or the number of dual variables

    # First compute EVD of preH and keep eigenvalues above a threshold
    # threshold = 1e-8 * ds
    threshold = 1e-5
    # println("eigenvalues:", eigvals(preH))
    vals, vecs = eigen(Symmetric(preH), threshold, Inf)
    n::Int64 = length(vals)
    # println("preH = ", preH)
    println("effective dimension of preH = ", n)
    preH::Array{Float64, 2} = Diagonal(vals)
    Cn = vecs' * Cn

    # if n == 0
    #     println("LP!")
    #     nd::Int = (m - 1) / 2
    #     (dw, y, new_lambda) = interior_point_LP.InteriorPoint(c, u, l, c0, m, nd, ds)
    #     return dw, y, new_lambda
    # end

    println("Non LP!")
    b::Array{Float64} = vcat(l, - u, - c0)
    # println("time to find initial point:")
    # x: primal varialbes; y: slack variables; lambda: dual variables
    (x, y, lambda) = InitialPoint(Cn, preH, c, b, m, nx, n)
    # for i = 1 : nx
    #     if x[i] < l[i]
    #         println("Error! initial x < l, i=", i, ", xi=", x[i], ", li=", l[i])
    #     end
    # end
    tau::Float64 = 0.95
    mu::Float64 = dot(y, lambda) / m
    iter::Int64 = 0
    while (mu > 1e-6 && iter < 10000)
        iter += 1
        # println("In IP, time to solve a linear system:")
        (dx_aff, dy_aff, dl_aff) = SolveLinearSysm(Cn, preH, c, b, x, y, lambda, 0.0, mu, m, nx, n)
        alpha_yaff::Float64 = SolveAlpha(dy_aff, - y, m)
        alpha_laff::Float64 = SolveAlpha(dl_aff, - lambda, m)
        alpha_aff::Float64 = min(alpha_yaff, alpha_laff)
        mu_aff::Float64 = dot(y + alpha_aff * dy_aff, lambda + alpha_aff * dl_aff) / m
        sigma::Float64 = (mu_aff / mu) ^ 3
        #println("In IP, time to solve a linear system:")
        (dx, dy, dl) = SolveLinearSysm(Cn, preH, c, b, x, y, lambda, sigma, mu, m, nx, n)
        tau = 1 - 0.9 * (1 - tau)
        alpha_pri::Float64 = SolveAlpha(dy, - tau * y, m)
        alpha_dual::Float64 = SolveAlpha(dl, - tau * lambda, m)
        alpha::Float64 = min(alpha_pri, alpha_dual)
        x = x + alpha * dx
        y = y + alpha * dy
        # for i = 1 : nx
        #     if x[i] - l[i] < 1e-5
        #         println("Error! middle x < l, i=", i, ", xi=", x[i], ", li=", l[i])
        #     end
        # end
        lambda = lambda + alpha * dl
        # for i = 1 : m
        #     if y[i] < - 1e-5
        #         println("Error! middle y < 0, i=", i, ", y[i]=", y[i])
        #     end
        #     if lambda[i] < - 1e-5
        #         println("Error! middle lambda < 0, i=", i, ", lambda[i]=", labmda[i])
        #     end
        # end

        mu = dot(y, lambda) / m
        if iter == 10000
            println("Max iteration number is reached!")
        end
    end
    return x, y, lambda
end

end
