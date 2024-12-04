function lagrange_interpolation(L, f)
    K = size(L, 1)
    if length(f) != K
        throw(ArgumentError("The length of f must match L."))
    end
    return sum(L[j, :] .* f[j] for j in 1:K)
end

function lagrange_basis(tau, t)
    K = length(tau)
    L = zeros(eltype(t), K, length(t))

    for j in 1:K
        L_j = ones(eltype(t), length(t))
        for m in 1:K
            if m != j
                L_j .*= (t .- tau[m]) / (tau[j] - tau[m])
            end
        end
        L[j, :] .= L_j
    end

    return L
end
