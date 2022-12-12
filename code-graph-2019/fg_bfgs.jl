module bfgs

using LinearAlgebra

# Compute trace inverse objective value with svd decomposition
# --------------------------------------------------------------------------------------------------------
function EvalObj(svdF::SVD, w::Array{Float64}, Cnx::Array{Float64, 2},
    nd::Int64, nr::Int64, n::Int64, sqrt_gamma::Array{Float64, 2}, vr::Float64,
    nt::Int64, nx::Int64)

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

    return sum(1 ./ (vr .+ eigenv))
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
function RunSVD(w::Array{Float64}, Fns::Array{Float64, 2}, Cnx::Array{Float64, 2},
    gamma::Array{Float64,2}, nx::Int64, nt::Int64, n::Int64, nd::Int64,
    nr::Int64, vr::Float64, svdF::SVD)
    # println("RunCG!")

    sqrt_gamma = sqrt(gamma)
    U, D = preM_direct(w, Cnx, svdF, sqrt_gamma, nd, nr, vr)
    preM::Array{Float64, 3} = zeros(nx, nt, n)
    for i::Int = 1 : n
        vecF = Fns[:, ((i - 1) * nt + 1) : (i * nt)]
        preM[:, :, i] = (vecF - U * D * U' * vecF) ./ vr
    end

    P2::Array{Float64, 4} = zeros(n, n, nt, nt)
    for i::Int = 1 : n
        for j::Int = 1 : i
            P2[i, j, :, :] = gamma \ preM[:, :, i]' * preM[:, :, j]
            P2[j, i, :, :] = gamma \ preM[:, :, j]' * preM[:, :, i]
        end
    end
    M1::Array{Float64, 2} = zeros(n, n)
    for i::Int = 1 : n
        for j::Int = 1 : i
            M1[i, j] = tr(P2[i, j, :, :])
            M1[j, i] = M1[i, j]
        end
    end
    return M1
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

# Compute the gradient of the objective (with SVD)
# --------------------------------------------------------------------------------------------------------
function EvalGradient(w::Array{Float64}, grad_f::Array{Float64,1},
    Fns::Array{Float64, 2}, Cnx::Array{Float64, 2}, gamma::Array{Float64,2},
    nx::Int64, nt::Int64, n::Int64, nd::Int64, nr::Int64, vr::Float64, svdF::SVD)

    M1 = RunSVD(w, Fns, Cnx, gamma, nx, nt, n, nd, nr, vr, svdF)
    grad_f[:] = GradientApprox(w, M1, Cnx, nd, nr)
end

end
