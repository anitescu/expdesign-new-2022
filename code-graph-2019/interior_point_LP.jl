module interior_point_LP
include("./fgh.jl")
using LinearAlgebra
# Implement the dogleg step (ignore the Hessian matrix in the SQP Algorithm)

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
# Solve for dx
function SolveDx(y::Array{Float64}, lambda::Array{Float64}, rhs::Array{Float64},
    m::Int64, nx::Int64)
    mu_const = lambda[m] / y[m]
    v0::Array{Float64} = ones(nx)

    v1::Array{Float64} = FindInvD(v0, y, lambda, nx)
    v2::Array{Float64} = FindInvD(rhs, y, lambda, nx)
    return v2 - v1 * dot(v0, v2) / (dot(v0, v1) + 1 / mu_const)
end

# --------------------------------------------------------------------------------------------------------
# Solve a linear system (16.58) for dx, dy and dl
function SolveLinearSysm(c::Array{Float64}, b::Array{Float64}, x::Array{Float64},
    y::Array{Float64}, lambda::Array{Float64}, sigma::Float64, mu::Float64,
    m::Int64, nx::Int64)

    rd::Array{Float64} = - ATx(lambda, nx, m) + c
    Ax::Array{Float64} = vcat(x, - x, - sum(x))
    rp::Array{Float64} = Ax - y - b
    vec::Array{Float64} = FindVec(rp, y, lambda, sigma, mu, m)
    rhs::Array{Float64} = - rd + ATx(vec, nx, m)
    dx::Array{Float64} = SolveDx(y, lambda, rhs, m, nx)
    dy::Array{Float64} = vcat(dx, - dx, - sum(dx)) + rp
    # v0::Array{Float64} = ones(m)
    v1::Array{Float64} = lambda./ y
    # dl::Array{Float64} = - lambda + sigma * mu * (v0./ y) - v1.* dy
    dl::Array{Float64} = - lambda + sigma * mu ./ y - v1.* dy

    ATlamb = ATx(dl, nx, m)
    test = - ATlamb + rd
    if norm(test) > 1e-3
        println("check norm = ", norm(test))
    end
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
function InitialPoint(c::Array{Float64}, b::Array{Float64}, m::Int64, nx::Int64)
    x0::Array{Float64} = zeros(nx)
    y0::Array{Float64} = 0.001 * ones(m)
    lambda0::Array{Float64} = 0.001 * ones(m)
    mu = dot(y0, lambda0) / m
    #println("In initial IP, time to solve a linear system:")
    (dx_aff, dy_aff, dl_aff) = SolveLinearSysm(c, b, x0, y0, lambda0, 0.0, mu, m, nx)
    for i::Int = 1 : m
        y0[i] = max(1, abs(y0[i] + dy_aff[i]))
        lambda0[i] = max(1, abs(lambda0[i] + dl_aff[i]))
    end
    return x0, y0, lambda0
end

# --------------------------------------------------------------------------------------------------------
# Implementation of Interior Point Method to Solve a QP (Algorithm 16.4 in Nocedal & Wright)
function InteriorPoint(c::Array{Float64}, u::Array{Float64}, l::Array{Float64}, c0::Float64,
    m::Int64, nx::Int64, ds::Float64)
    # nx is the dimension of the variable, not necessarily the mesh size
    # n is the dimension of preH, c*log(nx)
    # m is the number of constraints, or the number of dual variables

    b::Array{Float64} = vcat(l, - u, - c0)
    # println("time to find initial point:")
    # x: primal varialbes; y: slack variables; lambda: dual variables
    (x, y, lambda) = InitialPoint(c, b, m, nx)
    # for i = 1 : nx
    #     if x[i] < l[i]
    #         println("Error! initial x < l, i=", i, ", xi=", x[i], ", li=", l[i])
    #     end
    # end
    tau::Float64 = 0.95
    mu::Float64 = dot(y, lambda) / m
    iter::Int64 = 0
    while (mu > 1e-6 && iter < 10000)
        # println("mu = ", mu)
        iter += 1
        # println("In IP, time to solve a linear system:")
        (dx_aff, dy_aff, dl_aff) = SolveLinearSysm(c, b, x, y, lambda, 0.0, mu, m, nx)
        alpha_yaff::Float64 = SolveAlpha(dy_aff, - y, m)
        alpha_laff::Float64 = SolveAlpha(dl_aff, - lambda, m)
        alpha_aff::Float64 = min(alpha_yaff, alpha_laff)
        mu_aff::Float64 = dot(y + alpha_aff * dy_aff, lambda + alpha_aff * dl_aff) / m
        sigma::Float64 = (mu_aff / mu) ^ 3
        #println("In IP, time to solve a linear system:")
        (dx, dy, dl) = SolveLinearSysm(c, b, x, y, lambda, sigma, mu, m, nx)
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
