
using JuMP
import Ipopt
import Plots
using LinearAlgebra
using Printf
using Glob

foreach(include, glob("*.jl", "lib"))

# CGL, LGL, LGR, LG
const method = "LGR"

# supported by LGR and LG methods
const integral = false

# supported by LGR differentiation method
const costate = false

# number of grid points
const N = 200

#
# Earth constants
#

const µ🜨   = 3.986012e14   # m³/s²
const r🜨   = 6378145       # m
const rho0 = 1.225         # kg/m³
const H0   = 8500          # m
const g0   = 9.80665       # m/s²

#
# Vehicle constants
#

const Aref = 10
const Cd = 0.2
const Isp = 300
const mi = 5000
const mprop = 0.6*mi
const Tmax = mi*g0*2

#
# Initial conditions
#

const hi = 0
const vi = 0

#
# Scaling
#

const r_scale = norm(r🜨)
const v_scale = sqrt(µ🜨/r_scale)
const t_scale = r_scale / v_scale
const m_scale = mi
const a_scale = v_scale / t_scale
const f_scale = m_scale * a_scale
const area_scale = r_scale^2
const vol_scale = r_scale * area_scale
const d_scale = m_scale / vol_scale
const mdot_scale = m_scale / t_scale

#
# Applying scaling
#

const his = hi / r_scale
const vis = vi / v_scale
const mis = mi / m_scale
const r🜨s = r🜨 / r_scale
const mprops = mprop / m_scale
const Tmaxs = Tmax / f_scale
const H0s = H0 / r_scale
const rho0s = rho0 / d_scale
const Arefs = Aref / area_scale
const c = Isp*g0 / v_scale

#
# Computed values
#

const mfs = mis - mprops

function goddard()
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 5)
    set_optimizer_attribute(model, "print_user_options", "yes")
    set_optimizer_attribute(model, "max_iter", 2000)
    set_optimizer_attribute(model, "tol", 1e-12)

    tau, ptau, xtau, w, D, A, K, P = psmethod(method, N)

    #
    # Variables
    #

    @variable(model, 0 <= h[i=1:N] <= Inf, start=0)
    @variable(model, 0 <= v[i=1:N] <= Inf, start=0)
    @variable(model, mfs <= m[i=1:N] <= mis, start=mis)
    @variable(model, 0 <= T[i=1:K] <= Tmaxs, start=Tmaxs)
    @variable(model, 0 <= tf <= Inf, start=20/t_scale)

    #
    # Endpoint variable slices
    #

    xi = vcat(h[1], v[1], m[1])
    xf = vcat(h[N], v[N], m[N])

    #
    # Collocated variable slices
    #

    offset = 0
    if method == "LGR" || method == "LG"
        offset = 1
    end

    @expression(model, hc[i=1:K], h[i+offset])
    @expression(model, vc[i=1:K], v[i+offset])
    @expression(model, mc[i=1:K], m[i+offset])
    xc = hcat(hc, vc, mc)

    #
    # Polynomial variable slices
    #

    @expression(model, hp[i=1:P], h[i])
    @expression(model, vp[i=1:P], v[i])
    @expression(model, mp[i=1:P], m[i])
    xp = hcat(hp, vp, mp)

    #
    # Fixed constraints
    #

    fix(h[1], his, force=true)
    fix(v[1], vis, force=true)
    fix(m[1], mis, force=true)

    display(his)
    display(vis)
    display(mis)
    display(mfs)
    display(c)

    #
    # Dynamical constraints
    #

    D1 = 2.0 / tf * D
    A1 = tf / 2.0 * A
    w1 = w * tf ./ 2.0

    @expression(model, rho[i=1:K], rho0s*exp(-hc[i]/H0s))
    @expression(model, Drag[i=1:K], -0.5*Cd*Arefs*rho[i]*vc[i]^2)
    @expression(model, rsqr[i=1:K], (r🜨s + hc[i])^2)

    F = hcat(
              vc,
              -1.0 ./ rsqr + (T + Drag) ./ mc,
              -T ./ c,
             )

    if integral
        @constraint(model, xc == xi' .+ A1 * F)

        if method == "LG"
            @constraint(model, xf == xi + F' * w1)
        end
    else
        @constraint(model, D1 * xp == F)
        if method == "LG"
            @constraint(model, xf == xi + F' * w1)
        end
    end

    #
    # Objective
    #

    @objective(model, Max, h[N])

    #
    # Solve the problem
    #

    solve_and_print_solution(model)

    #
    # Descale and interpolate the variables
    #

    range = LinRange(-1,1,20)
    L = lagrange_basis(ptau, range)

    h = lagrange_interpolation(L, value.(hp)) * r_scale
    v = lagrange_interpolation(L, value.(vp)) * v_scale
    m = lagrange_interpolation(L, value.(mp)) * m_scale

    L = lagrange_basis(tau, range)

    T = lagrange_interpolation(L, value.(T)) * f_scale

    #
    # Construct ranges of real times at the interpolation points
    #

    tf = value(tf)
    t = (range * tf / 2.0 .+ tf / 2.0) * t_scale

    #
    # Do some plotting of interpolated results
    #

    p1 = Plots.plot(
                    t,
                    h,
                    xlabel = "Time (s)",
                    ylabel = "Height",
                    legend = false
                   )
    p2 = Plots.plot(
                    t,
                    v,
                    xlabel = "Time (s)",
                    ylabel = "Velocity",
                    legend = false
                   )
    p3 = Plots.plot(
                    t,
                    m,
                    xlabel = "Time (s)",
                    ylabel = "Mass",
                    legend = false
                   )
    p4 = Plots.plot(
                    t,
                    T,
                    xlabel = "Time (s)",
                    ylabel = "Thrust",
                    legend = false
                   )
    display(Plots.plot(p1, p2, p3, p4, layout=(2,2), legend=false))
    readline()

    @assert is_solved_and_feasible(model)
end

goddard()
