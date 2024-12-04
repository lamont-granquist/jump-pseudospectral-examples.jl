function rv2oe(μ, r, v)
    rmag = sqrt(sum(i^2 for i in r))         # Magnitude of position vector
    vmag = sqrt(sum(i^2 for i in v))         # Magnitude of velocity vector
    rhat = r ./ rmag                         # Unit vector in the direction of r
    hv = cross(r, v)                         # Specific angular momentum vector
    hhat = hv ./ sqrt(sum(i^2 for i in hv))  # Unit vector of angular momentum
    eccvec = cross(v / μ, hv) - rhat         # Eccentricity vector
    sma = 1.0 / (2.0 / rmag - vmag^2 / μ)    # Semi-major axis
    l = sum(i^2 for i in hv) / μ             # Semi-latus rectum

    # Parameters for frame transformation
    d = 1.0 + hhat[3]
    p = d == 0 ? 0 : hhat[1] / d
    q = d == 0 ? 0 : -hhat[2] / d
    const1 = 1.0 / (1.0 + p^2 + q^2)

    fhat = [
        const1 * (1.0 - p^2 + q^2),
        const1 * 2.0 * p * q,
        -const1 * 2.0 * p
    ]

    ghat = [
        const1 * 2.0 * p * q,
        const1 * (1.0 + p^2 - q^2),
        const1 * 2.0 * q
    ]

    # Calculate Keplerian elements
    h = dot(eccvec, ghat)
    xk = dot(eccvec, fhat)
    x1 = dot(r, fhat)
    y1 = dot(r, ghat)
    xlambdot = atan(y1, x1)                       # True longitude
    ecc = sqrt(h^2 + xk^2)                        # Eccentricity
    inc = 2.0 * atan(sqrt(p^2 + q^2))             # Inclination
    lan = inc > eps() ? atan(p, q) : 0.0          # Longitude of ascending node
    argp = ecc > eps() ? atan(h, xk) - lan : 0.0  # Argument of periapsis
    nu = xlambdot - lan - argp                    # True anomaly

    # Normalize angles to [0, 2π]
    lan = mod2pi(lan)
    argp = mod2pi(argp)
    nu = mod2pi(nu)

    return sma, ecc, inc, lan, argp, nu, l
end
