using LinearAlgebra

# Uses barycentric lagrange interpolation to compute the
# differentation matrix based on the collocation points in tau
#
# Berrut JP, Trefethen LN. Barycentric lagrange interpolation. SIAM review. 2004;46(3):501-17.
function dmatrix(tau)
    N = length(tau)

    X = repeat(tau, 1, N)
    Xdiff = X - X' + I
    W = repeat(1.0 ./ prod(Xdiff, dims=2), 1, N)

    D = W ./ (W' .* Xdiff)
    D[diagind(D)] = 1 .- sum(D, dims=1)
    D = -D'

    return D
end
