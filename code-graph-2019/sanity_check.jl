using Pkg
using LinearAlgebra
using Plots
pyplot()

# Solution to advection-diffusion equation
# --------------------------------------------------------------------------------------------------------
function sol(c1::Float64, c2::Float64, mu::Float64, x1::Float64, x2::Float64,
    y1::Float64, y2::Float64, t::Float64, p::Int64)
    # (x1, x2) are from output domain, and (y1, y2) are from input domain
    sol = 0
    for k1::Int64 = 0 : p
        b1 = k1 % 2
        for k2::Int64 = 0 : p
            b2 = k2 % 2
            tmp = 0
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
function pto(c1::Float64, c2::Float64, mu::Float64, T::Float64, nx_each::Int64, nt::Int64,
    nd::Int64, nr::Int64, p::Int64)
    # we implicitly assume R = 1
    R::Float64 = 1.0
    nx::Int = nx_each * nx_each
    dx::Float64 = 2 / nx_each
    ds::Float64 = dx * dx
    F::Array{Float64,2} = zeros(nd * nr * nt, nx)
    for i1::Int = 1 : nd
        for i2::Int = 1 : nr
            x1 = i2 * (R / nr) * cos(i1 * 2 * pi / nd)
            x2 = i2 * (R / nr) * sin(i1 * 2 * pi / nd)
            i = (i1 - 1) * nr + i2
            for j::Int = 1 : nx
                y1 = (floor((j - 1) / nx_each) + 0.5) * dx - 1
                y2 = (mod(j - 1, nx_each) + 0.5) * dx - 1
                for k::Int64 = 1 : nt
                    t = (k / nt) * T
                    F[(i - 1) * nt + k, j] = sol(c1, c2, mu, x1, x2, y2, y1, t, p) * ds
                end
            end
        end
    end
    return F
    #return Array(Symmetric(F))
end

# Define Input Function
# --------------------------------------------------------------------------------------------------------
function Input(x::Float64, y::Float64)
    return sin(x * pi) * sin(y * pi)
    # return exp(abs(x) + abs(y))
end

# Create Input Vector
# --------------------------------------------------------------------------------------------------------
function InputVector(nx_each::Int64)
    vec = zeros(nx_each * nx_each)
    dx = 2 / nx_each
    for i::Int = 1 : nx_each
        y = (i - 0.5) * dx - 1
        for j::Int = 1 : nx_each
            x = (j - 0.5) * dx - 1
            vec[(i - 1) * nx_each + j] = Input(x, y)
        end
    end
    return vec
end

function sanity_plot(input::Array{Float64}, output::Array{Float64}, nx_each::Int,
    nd::Int, nr::Int)
    # Gadfly.spy(reshape(input, nx_each, nx_each)')
    x = (1 : nx_each) / nx_each
    y = (1 : nx_each) / nx_each
    z = reshape(input, nx_each, nx_each)
    heatmap(x, y, z, aspect_ratio = 1)
    savefig("input.pdf")
    theta = LinRange(2 * pi / nr, 2 * pi, nr)
    r = LinRange(1 / nd, 1, nd)
    out = reshape(output, nd, nd)
    heatmap(theta, r, out, proj = :polar)
    savefig("output.pdf")
end

function sanity_check_main()
    nx_each::Int = 30
    nd = nx_each
    nr = nx_each

    c1::Float64 = 1.0
    c2::Float64 = 0.0
    mu::Float64 = 0.5
    p::Int = 5
    T::Float64 = 0.0
    nt::Int = 1

    input::Array{Float64} = InputVector(nx_each)
    F = pto(c1, c2, mu, T, nx_each, nt, nd, nr, p)
    output = F * input
    sanity_plot(input, output, nx_each, nd, nr)
    return input, output
end
