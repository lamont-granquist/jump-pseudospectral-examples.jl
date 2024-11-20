using FastGaussQuadrature
using FastTransforms

include("dmatrix.jl")

function psmethod(method, N)
    A = []

    if method == "LGL"
        P = N
        K = N
        tau, w = gausslobatto(K)
        D = dmatrix(tau)
        xtau = tau
    elseif method == "LGR"
        P = N
        K = N-1
        tau, w = gaussradau(K)
        tau = reverse(-tau)
        w = reverse(w)
        tau = [-1; tau]
        D = dmatrix(tau)
        D = D[2:end, :]
        tau = tau[2:end]
        D1N = D[:, 2:end]
        A = inv(D1N)
        xtau = [ -1; tau ]
    elseif method == "LG"
        P = N-1
        K = N-2
        tau, w = gausslegendre(K)
        tau = [-1; tau]
        D = dmatrix(tau)
        D = D[2:end, :]
        tau = tau[2:end]
        xtau = [ -1; tau; 1 ]
    elseif method == "CGL"
        P = N
        K = N
        tau = reverse(clenshawcurtisnodes(Float64, N))
        μ = FastTransforms.chebyshevmoments1(Float64, N)
        w = clenshawcurtisweights(μ)
        D = dmatrix(tau)
        xtau = tau
    else
        error("invalid pseudospectral method")
    end

    return tau, xtau, w, D, A, K, P
end
