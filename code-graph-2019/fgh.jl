module fgh
using LinearAlgebra

# --------------------------------------------------------------------------------------------------------
# Define the integrand in the integral equation
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

# --------------------------------------------------------------------------------------------------------
# Define interpolation matrix
function InterpolationMatrix(n::Int64, nx::Int64, nd::Int64, nr::Int64,
    nt::Int64, c1::Float64, c2::Float64, mu::Float64, T::Float64, p::Int64)
    # n_each = interpolation points on each side
    # nx_each = discretization points on each side, nx >> n.
    # c1, c2: constant in PDE, not to confuse with c in IP algorithm
    n_each::Int = sqrt(n)
    nx_each::Int = sqrt(nx)
    Fn::Array{Float64, 2} = zeros(n * nt, n)
    for i::Int = 0 : (n - 1)
        # i correspondes to the output domain
        i1::Int = floor(i / n_each)
        i2::Int = i % n_each
        # row, or y axis
        s1::Float64  = cos(i1 * pi / (n_each - 1))
        # column, or x axis
        s2::Float64  = cos(i2 * pi / (n_each - 1))
        for j::Int = 0 : (n - 1)
            # j correspondes to the input domain
            j1::Int = floor(j / n_each)
            j2::Int = j % n_each
            t1::Float64  = cos(j1 * pi / (n_each - 1))
            t2::Float64  = cos(j2 * pi / (n_each - 1))
            for k::Int = 1 : nt
                t = k * (T / nt)
                Fn[i * nt + k, j + 1] = sol(c1, c2, mu, s2, s1, t2, t1, t, p) * (2 / nx_each) ^ 2
            end
        end
    end

    # xn is the set of interpolation points in one dimension
    xn::Array{Float64} = zeros(n_each)
    for i::Int = 0 : (n_each - 1)
        xn[i + 1] = cos(i * pi / (n_each - 1))
    end

    # Cnx is the coefficient matrix for the output domain
    Cnx::Array{Float64, 2} = zeros(n, nr * nd)
    # unit circle as the output domain: R = 1.0; theta is in range (0, 2pi)
    R::Float64 = 1.0
    for i::Int = 1 : nd
        for j::Int = 1 : nr
            xtmp::Float64 = j * (R / nr) * cos(i * 2 * pi / nd)
            ytmp::Float64 = j * (R / nr) * sin(i * 2 * pi / nd)
            xc::Array{Float64} = ones(n_each)
            yc::Array{Float64} = ones(n_each)
            for k::Int = 1 : n_each
                for l::Int = [1 : (k - 1); (k + 1) : n_each]
                    xc[k] *= (xtmp - xn[l]) / (xn[k] - xn[l])
                    yc[k] *= (ytmp - xn[l]) / (xn[k] - xn[l])
                end
            end
            for k::Int = 0 : (n_each - 1)
                for l::Int = 1 : n_each
                    Cnx[k * n_each + l, (i - 1) * nr + j] = xc[l] * yc[k + 1]
                end
            end
        end
    end

    # Cny is the coefficient matrix for the input domain
    Cny_each::Array{Float64, 2} = ones(n_each, nx_each)
    for i::Int = 1 : nx_each
        x::Float64 = - 1 + (i - 0.5) * (2 / nx_each)
        for j::Int = 1 : n_each
            for k::Int in [1 : (j - 1); (j + 1) : n_each]
                Cny_each[j, i] *= (x - xn[k]) / (xn[j] - xn[k])
            end
        end
    end

    Cny::Array{Float64, 2} = zeros(n, nx)
    for i::Int = 0 : (nx - 1)
        # row, or y axis
        i1::Int = floor(i / nx_each)
        # column, or x axis
        i2::Int = i % nx_each
        for j::Int = 0 : (n - 1)
            j1::Int = floor(j / n_each)
            j2::Int = j % n_each
            # be careful here
            Cny[j + 1, i + 1] = Cny_each[j1 + 1, i1 + 1] * Cny_each[j2 + 1, i2 + 1]
        end
    end

    Fns = Cny' * Fn'
    return Fns, Cnx
end

# --------------------------------------------------------------------------------------------------------
# Conjugate gradient
function MatrixVecMult(w::Array{Float64}, Fns::Array{Float64, 2},
    Cnx::Array{Float64, 2}, x::Array{Float64}, nx::Int64, nt::Int64, n::Int64,
    nd::Int64, nr::Int64, gamma::Array{Float64, 2}, vr::Float64)
    # note w is of size nd
    y::Array{Float64} = x
    x = Fns' * x # after multiplication, x has size n * nt
    z = zeros(nd * nr * nt)
    for i1::Int = 1 : nd
        for i2::Int = 1 : nr
            i = (i1 - 1) * nr + i2
            for j::Int = 1 : n
                z[((i - 1) * nt + 1) : i * nt] += Cnx[j, i] * (gamma \ x[((j - 1) * nt + 1) : j * nt])
            end
            z[((i - 1) * nt + 1) : i * nt] *= w[i1]
        end
    end
    x = zeros(n * nt)
    for i::Int = 1 : n
        for j1::Int = 1 : nd
            for j2::Int = 1 : nr
                j = (j1 - 1) * nr + j2
                x[((i - 1) * nt + 1) : i * nt] += Cnx[i, j] * z[((j - 1) * nt + 1) : j * nt]
            end
        end
    end
    x = Fns * x
    return x + vr * y
end

# --------------------------------------------------------------------------------------------------------
# Conjugate gradient
function CG(w::Array{Float64}, Fns::Array{Float64, 2},
    Cnx::Array{Float64, 2}, k::Int64, nx::Int64, nt::Int64, n::Int64, nd::Int64,
    nr::Int64, gamma::Array{Float64, 2}, vr::Float64)
    # println("CG!")
    ret::Array{Float64} = zeros(nx, nt)
    for i::Int = 1 : nt
        x::Array{Float64} = zeros(nx)
        r::Array{Float64} = - Fns[:, (k - 1) * nt + i]
        p = - r
        # println(norm(r))
        while (norm(r) > 1e-8)
            #println(norm(r))
            q::Array{Float64} = MatrixVecMult(w, Fns, Cnx, p, nx, nt, n, nd, nr, gamma, vr)
            c::Float64 = dot(r, r)
            a::Float64 = c / dot(p, q)
            x = x + a * p
            r = r + a * q
            b::Float64 = dot(r, r) / c
            p = - r + b * p
        end
        ret[:, i] = x
    end
    return ret
end

# --------------------------------------------------------------------------------------------------------
# Compute preM with SVD decomposition, rather than CG
function preM_direct(w::Array{Float64}, Cnx::Array{Float64, 2}, svdF::SVD,
    sqrt_gamma::Array{Float64, 2}, nd::Int64, nr::Int64, vr::Float64)
    sqrt_w = sqrt.(abs.(w))
    Cnxw = similar(Cnx)
    for i::Int = 1 : nd
        Cnxw[:, ((i - 1) * nr + 1) : i * nr] = Cnx[:, ((i - 1) * nr + 1) : i * nr] * sqrt_w[i]
    end
    wCnx = kron(Cnxw, sqrt_gamma)
    svdWC = svd(wCnx)
    eigenM = Diagonal(svdF.S) * svdF.Vt * svdWC.U * Diagonal(svdWC.S .^ 2) * svdWC.U' * svdF.Vt' * Diagonal(svdF.S)
    vals, vecs = eigen(Symmetric(eigenM))
    U = svdF.U * vecs
    D = vals ./ (vals .+ vr)
    return U, Diagonal(D)
end

# --------------------------------------------------------------------------------------------------------
# Perform Conjugate gradient
function RunCG(w::Array{Float64}, Fns::Array{Float64, 2}, Cnx::Array{Float64, 2},
    gamma::Array{Float64,2}, nx::Int64, nt::Int64, n::Int64, nd::Int64,
    nr::Int64, vr::Float64, svdF::SVD)
    # println("RunCG!")
    # preM1::Array{Float64, 3} = zeros(nx, nt, n)
    # for i::Int = 1 : n
    #     preM1[:, :, i] = CG(w, Fns, Cnx, i, nx, nt, n, nd, nr, gamma, vr)
    # end

    sqrt_gamma = sqrt(gamma)
    U, D = preM_direct(w, Cnx, svdF, sqrt_gamma, nd, nr, vr)
    preM::Array{Float64, 3} = zeros(nx, nt, n)
    for i::Int = 1 : n
        vecF = Fns[:, ((i - 1) * nt + 1) : (i * nt)]
        preM[:, :, i] = (vecF - U * D * U' * vecF) ./ vr
    end
    # println("diff: ", norm(preM1 - preM))

    P1::Array{Float64, 4} = zeros(n, n, nt, nt)
    P2::Array{Float64, 4} = zeros(n, n, nt, nt)
    for i::Int = 1 : n
        for j::Int = 1 : i
            P1[i, j, :, :] = gamma \ preM[:, :, i]' * Fns[:, ((j - 1) * nt + 1) : j * nt]
            P1[j, i, :, :] = gamma \ preM[:, :, j]' * Fns[:, ((i - 1) * nt + 1) : i * nt]
            P2[i, j, :, :] = gamma \ preM[:, :, i]' * preM[:, :, j]
            P2[j, i, :, :] = gamma \ preM[:, :, j]' * preM[:, :, i]
        end
    end
    M1::Array{Float64, 2} = zeros(n, n)
    M2::Array{Float64, 2} = zeros(n, n)
    for i::Int = 1 : n
        for j::Int = 1 : i
            M1[i, j] = tr(P2[i, j, :, :])
            M1[j, i] = M1[i, j]
            M2[i, j] = tr(P1[i, j, :, :] * P2[j, i, :, :]) + tr(P1[j, i, :, :] * P2[i, j, :, :])
            M2[j, i] = M2[i, j]
        end
    end
    return M1, M2
end

# --------------------------------------------------------------------------------------------------------
# Compute gradient by interpolation
function GradientApprox(w::Array{Float64}, M1::Array{Float64, 2},
    Cnx::Array{Float64, 2}, nd::Int64, nr::Int64)
    grad::Array{Float64} = zeros(nd)
    for i::Int = 1 : nd
        for j::Int = 1 : nr
            k = (i - 1) * nr + j
            grad[i] += - Cnx[:, k]' * M1 * Cnx[:, k]
        end
    end
    return grad
end

# --------------------------------------------------------------------------------------------------------
# Compute hessian_vector product by interpolation
# function HessianApprox(w::Array{Float64}, x::Array{Float64}, M1::Array{Float64, 2},
#     M2::Array{Float64, 2}, Cnx::Array{Float64, 2})
#     x = Cnx * x
#     x = M2 * x
#     x = Cnx' * x
#     return x
# end

# function HessianApprox(w::Array{Float64}, x::Array{Float64}, M1::Array{Float64, 2},
#     M2::Array{Float64, 2}, Cn::Array{Float64, 2})
#     Hx::Array{Float64} = zeros(nx)
#     newM1::Array{Float64, 2} = zeros(n, nx)
#     newM2::Array{Float64, 2} = zeros(n, nx)
#     for i::Int = 1 : nx
#         newM1[:, i] = M1 * Cn[:, i] * x[i]
#         newM2[:, i] = M2 * Cn[:, i]
#     end
#     preH::Array{Float64, 2} = zeros(n, n)
#     for i::Int = 1 : n
#         for j::Int = 1 : n
#             preH[i, j] = dot(newM1[i, :], newM2[j, :])
#         end
#     end
#     for i::Int = 1 : nx
#         Hx[i] = dot(Cn[:, i], preH * Cn[:, i])
#     end
#     return 2 * Hx
# end
#
end
