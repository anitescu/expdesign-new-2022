module mymodule
using SparseArrays
using LinearAlgebra

# Solution to advection-diffusion equation
# --------------------------------------------------------------------------------------------------------
function sol(c1::Float64, c2::Float64, mu::Float64, x1::Float64, x2::Float64,
    y1::Float64, y2::Float64, t::Float64, p::Int64)
    # (x1, x2) on output domain, and (y1, y2) on input domain
    sol::Float64 = 0.0
    for k1::Int64 = 0 : p
        b1 = k1 % 2
        for k2::Int64 = 0 : p
            b2 = k2 % 2
            tmp = 0.0
            if b1 == 1 && b2 == 1
                tmp = exp(- (c1 * y1 + c2 * y2) / (2 * mu)) * cos(k1 * pi * x1 / 2) * cos(k2 * pi * x2 / 2) * cos(k1 * pi * y1 / 2) * cos(k2 * pi * y2 / 2)
            elseif b1 == 1 && b2 == 0
                tmp = exp(- (c1 * y1 + c2 * y2) / (2 * mu)) * cos(k1 * pi * x1 / 2) * sin(k2 * pi * x2 / 2) * cos(k1 * pi * y1 / 2) * sin(k2 * pi * y2 / 2)
            elseif b1 == 0 && b2 == 1
                tmp = exp(- (c1 * y1 + c2 * y2) / (2 * mu)) * sin(k1 * pi * x1 / 2) * cos(k2 * pi * x2 / 2) * sin(k1 * pi * y1 / 2) * cos(k2 * pi * y2 / 2)
            else
                tmp = exp(- (c1 * y1 + c2 * y2) / (2 * mu)) * sin(k1 * pi * x1 / 2) * sin(k2 * pi * x2 / 2) * sin(k1 * pi * y1 / 2) * sin(k2 * pi * y2 / 2)
            end
            tmp *= exp(- mu * t * (k1 ^ 2 + k2 ^ 2) * pi ^ 2 / 4)
            sol += tmp
        end
    end
    return sol * exp(- (c1 ^ 2 + c2 ^ 2) * t / (4 * mu) + (c1 * x1 + c2 * x2) / (2 * mu))
end

# Define parameter-to-observable mapping
# --------------------------------------------------------------------------------------------------------
function pto(c1::Float64, c2::Float64, mu::Float64, T::Float64, nx_each::Int64,
    nt::Int64, nd::Int64, nr::Int64, p::Int64)
    # output domain: the unit circle on a 2d plane
    # input domain: rectangle [-1, 1] * [-1, 1]
    R::Float64 = 1.0
    nx::Int = nx_each * nx_each
    dx::Float64 = 2 / nx_each
    ds::Float64 = dx * dx
    F::Array{Float64,2} = zeros(nd * nr * nt, nx)
    # (x1, x2) output domain, (y2, y1) input domain
    for i1::Int = 1 : nd
        for i2::Int = 1 : nr
            x1 = i2 * (R / nr) * cos(i1 * 2 * pi / nd)
            x2 = i2 * (R / nr) * sin(i1 * 2 * pi / nd)
            i = (i1 - 1) * nr + i2
            for j::Int = 1 : nx
                # y1 corresponds to y axis on input domain, y2 corresponds to x axis
                y1 = (floor((j - 1) / nx_each) + 0.5) * dx - 1
                y2 = (mod(j - 1, nx_each) + 0.5) * dx - 1
                for k::Int = 1 : nt
                    t = (k / nt) * T
                    F[(i - 1) * nt + k, j] = sol(c1, c2, mu, x1, x2, y2, y1, t, p) * ds
                end
            end
        end
    end
    return F
    #return Array(Symmetric(F))
end

# --------------------------------------------------------------------------------------------------------
function MatrixVecMult(w::Array{Float64}, F::Array{Float64, 2}, x::Array{Float64},
    nx_each::Int64, nt::Int64, nd::Int64, nr::Int64, gamma::Array{Float64, 2}, vr::Float64)
    nx::Int = nx_each * nx_each
    y::Array{Float64} = x
    x = F * x
    z::Array{Float64} = zeros(nd * nr * nt)
    for i1::Int = 1 : nd
        for i2::Int = 1 : nr
            i::Int = (i1 - 1) * nr + i2
            z[((i - 1) * nt + 1) : i * nt] = w[i1] * (gamma \ x[((i - 1) * nt + 1) : i * nt])
        end
    end
    x = F' * z
    return x + vr * y
end

# Conjugate gradient
# --------------------------------------------------------------------------------------------------------
function CG(w::Array{Float64}, F::Array{Float64, 2}, k::Int64, nx_each::Int64,
    nt::Int64, nd::Int64, nr::Int64, gamma::Array{Float64, 2}, vr::Float64)
    # println("CG!")
    nx = nx_each * nx_each
    ret::Array{Float64, 2} = zeros(nx, nt)
    for i::Int = 1 : nt
        x::Array{Float64} = zeros(nx)
        r::Array{Float64} = - F[(k - 1) * nt + i, :]
        # println("size of r: ", size(r))
        p = - r
        # println(norm(r))
        while (norm(r) > 1e-8)
            # println("residual norm = ", norm(r))
            q::Array{Float64} = MatrixVecMult(w, F, p, nx_each, nt, nd, nr, gamma, vr)
            c::Float64 = dot(r, r)
            a::Float64 = c / dot(p, q)
            x = x + a * p
            r = r + a * q
            b::Float64 = dot(r, r) / c
            p = - r + b * p
        end
        ret[:, i] = x
    end
    #println("residual norm = ", norm(r))
    return ret
end

# Compute inv(F^TWF + I) * F
# --------------------------------------------------------------------------------------------------------
function computeM(w::Array{Float64}, nx_each::Int64, nt::Int64, vr::Float64,
    F::Array{Float64, 2}, nd::Int64, nr::Int64, gamma::Array{Float64, 2})
    nx::Int = nx_each * nx_each
    M::Array{Float64, 3} = zeros(nd * nr, nx, nt)
    for k1::Int = 1 : nd
        for k2::Int = 1 : nr
            k = (k1 - 1) * nr + k2
            M[k, :, :] = CG(w, F, k, nx_each, nt, nd, nr, gamma, vr)
        end
    end
    return M
end

# --------------------------------------------------------------------------------------------------------
function dertrace(M::Array{Float64, 3}, gamma::Array{Float64, 2}, nd::Int64, nr::Int64)
    D::Array{Float64} = zeros(nd)
    for k1::Int = 1 : nd
        for k2::Int = 1 : nr
            k = (k1 - 1) * nr + k2
            D[k1] += - tr(gamma \ M[k,:,:]' * M[k,:,:])
        end
    end
    return D
end

# Check Gradient of the objective function without CG
# --------------------------------------------------------------------------------------------------------
function sparsetr(F::Array{Float64,2}, w::Array{Float64,1}, nx_each::Int64,
    nt::Int64, nd::Int64, nr::Int64, gamma::Array{Float64,2}, vr::Float64)
    # Note gamma is the covariance matrix in time, of size nt by nt
    nx = nx_each * nx_each
    M = zeros(nx, nx)
    # P = blockdiag()
    # for i1::Int = 1 : nd
    #     for i2::Int = 1 : nr
    #         i = (i1 - 1) * nr + i2
    #         P = blockdiag(P, sparse(w[i1] * inv(gamma)))
    #     end
    # end
    precision = inv(gamma)
    P = kron(Diagonal(w), Matrix{Float64}(I, nr, nr), precision)
    eye = Matrix{Float64}(I, nx, nx)
    # print(F, size(P), nx)
    M = Symmetric(transpose(F) * P * F + vr * eye)
    # println("eigenvalues of M: ", eigvals(M))
    return M
end

# --------------------------------------------------------------------------------------------------------
function prod(F::Array{Float64,2}, w::Array{Float64,1}, nx_each::Int64,
    nt::Int64, nd::Int64, nr::Int64, gamma::Array{Float64,2}, vr::Float64)
    # println("current w: ", w)
    nx = nx_each * nx_each
    M = sparsetr(F, w, nx_each, nt, nd, nr, gamma, vr)
    chol = cholesky(M)
    # prod_arr is an analogy to M in the computeM function
    prod_arr = zeros(nd * nr, nx, nt)
    for i1::Int = 1 : nd
        for i2::Int = 1 : nr
            i = (i1 - 1) * nr + i2
            fi = transpose(F[((i - 1) * nt + 1) : i * nt, :])
            prod_arr[i, :, :] = chol.U \ (chol.L \ fi)
        end
    end
    return prod_arr
end

# --------------------------------------------------------------------------------------------------------
function dertrace2(prod_arr::Array{Float64,3}, nd::Int64, nr::Int64, gamma::Array{Float64,2})
    # D records the gradient vector
    D = zeros(nd)
    # The following loop computes the gradient
    for i1::Int = 1 : nd
        for i2::Int = 1 : nr
            i = (i1 - 1) * nr + i2
            vi = prod_arr[i, :, :]
            D[i1] += - tr(gamma \ transpose(vi) * vi)
        end
    end
    # println(D)
    return D
end

# Compute the objective function
# --------------------------------------------------------------------------------------------------------
function myfun1(w::Array{Float64,1}, F::Array{Float64,2}, nx_each::Int64, nt::Int64,
    nd::Int64, nr::Int64, gamma::Array{Float64,2}, vr::Float64)
    nx = nx_each * nx_each
    # P = blockdiag()
    # for i1::Int = 1 : nd
    #     for i2::Int = 1 : nr
    #         P = blockdiag(P, sparse(w[i1] * precision))
    #     end
    # end
    precision = inv(gamma)
    P = kron(Diagonal(w), Matrix{Float64}(I, nr, nr), precision)
    # eye = Matrix{Float64}(I, nx, nx)
    # Mat = Symmetric(transpose(F) * P * F + vr * eye)
    # # obj = tr(inv(Mat))
    # d = eigvals(Mat)
    # # println("d: ", d)
    # obj = sum(1 ./ d)

    svdF = svd(F)
    Mat2 = Diagonal(svdF.S) * svdF.U' * P * svdF.U * Diagonal(svdF.S)
    svdM = svd(Mat2)
    obj2 = sum(1 ./ (svdM.S .+ vr))
    # println("diff in obj: ", norm(obj - obj2))
    return obj2
end

# Compute the gradient of the objective (with CG)
# --------------------------------------------------------------------------------------------------------
function myfun2(w::Array{Float64,1}, grad_f::Array{Float64,1},
    F::Array{Float64,2}, nx_each::Int64, nt::Int64, nd::Int64, nr::Int64,
    gamma::Array{Float64,2}, vr::Float64)
    # M = computeM(w, nx_each, nt, vr, F, nd, nr, gamma)
    # old_grad_f = dertrace(M, gamma, nd, nr)
    M = prod(F, w, nx_each, nt, nd, nr, gamma, vr)
    grad_f[:] = dertrace2(M, nd, nr, gamma)
    # println("grad 1:", old_grad_f)
    # println("grad 2:", grad_f[:])
end

# Define hessian of the objective
# --------------------------------------------------------------------------------------------------------
function myfun3(w, mode, rows, cols, obj_factor, lambda, values, F, nx_each, nt, nd, nr, gamma, vr)
    if mode == :Structure
        # Symmetric matrix, fill the lower left triangle only
        idx = 1
        for row = 1 : nd
            for col = 1 : row
                rows[idx] = row
                cols[idx] = col
                idx += 1
            end
        end
    else
        # M::Array{Float64, 3} = computeM(w, nx_each, nt, F, nd, nr, gamma)
        M::Array{Float64, 3} = prod(F, w, nx_each, nt, nd, nr, gamma, vr)
        H::Array{Float64, 2} = zeros(nd * nr, nd * nr)
        for i = 1 : (nd * nr)
            for j = 1 : i
                H[i, j] = tr(gamma \ M[j,:,:]' * F[((i - 1) * nt + 1): i * nt, :]' * (gamma \ M[i,:,:]' * M[j,:,:])) + tr(gamma \ M[i,:,:]' * F[((j - 1) * nt + 1): j * nt, :]' * (gamma \ M[j,:,:]' * M[i,:,:]))
                H[j, i] = H[i, j]
            end
        end
        for i = 1 : nd
            for j = 1 : i
                val = sum(sum(H[((i - 1) * nr + 1) : i * nr, ((j - 1) * nr + 1) : j * nr]))
                k = Int(i * (i - 1) / 2 + j)
                values[k] = obj_factor * val
            end
        end
    end
end

# Implement Sum-up Rounding Strategy
# --------------------------------------------------------------------------------------------------------
function intapprox(nx::Int64, w::Array{Float64,1})
    # note nx = size(w), does not have to be the mesh size
    intw = zeros(nx)
    vec1 = zeros(nx)
    vec2 = ones(nx)
    for i = 1 : nx
        vec1[i] = 1
        v = dot(w, vec1) - dot(intw, vec2)
        if (v >= 0.5)
            intw[i] = 1
        end
    end
    return intw
end


# --------------------------------------------------------------------------------------------------------
function checkGap(w::Array{Float64}, lambda::Array{Float64}, F::Array{Float64, 2},
    nx::Int64, nt::Int64, nd::Int64, nr::Int64, gamma::Array{Float64, 2}, vr::Float64)
    nx_each = Int(sqrt(nx))
    if length(lambda) != 2 * nd + 1
        println("Length Error!")
    end
    mu::Float64 = lambda[end]
    # M = computeM(w, nx_each, nt, F, nd, nr, gamma)
    M::Array{Float64, 3} = prod(F, w, nx_each, nt, nd, nr, gamma, vr)
    D::Array{Float64} = dertrace2(M, nd, nr, gamma)
    v::Array{Float64} = zeros(nd)
    for i::Int = 1 : nd
        v[i] = D[i] + mu - lambda[i] + lambda[i + nd]
    end
    return(norm(v, Inf))
end

end
