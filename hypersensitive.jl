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
const integral = true

# supported by LGR differentiation method
const costate = false

# number of grid points
const N = 50

#
# Problem constants
#

const ti = 0
const tf = 1000
const xival = 1.5
const xfval = 1

const xmax = 50
const xmin = -xmax
const umax = 50
const umin = -umax

function hypersensitive()
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 5)
    set_optimizer_attribute(model, "print_user_options", "yes")
    set_optimizer_attribute(model, "max_iter", 2000)
    set_optimizer_attribute(model, "tol", 1e-12)

    tau, ptau, xtau, w, D, A, K, P = psmethod(method, N)

    #
    # Variables
    #

    @variable(model, xmin <= x[i=1:N] <= xmax, start=0)
    @variable(model, umin <= u[i=1:K] <= umax, start=0)

    #
    # Endpoint variable slices
    #

    xi = vcat(x[1])
    xf = vcat(x[N])

    #
    # Collocated variable slices
    #

    offset = 0
    if method == "LGR" || method == "LG"
        offset = 1
    end

    @expression(model, xc[i=1:K], x[i+offset])

    #
    # Polynomial variable slices
    #

    @expression(model, xp[i=1:P], x[i])

    #
    # Endpoint constraints
    #

    fix(x[1], xival, force=true)
    fix(x[N], xfval, force=true)

    #
    # Dynamical constraints
    #

    D1 = 2.0 / (tf - ti) * D
    A1 = (tf - ti) / 2.0 * A
    w1 = w * (tf - ti) ./ 2.0

    F = hcat(
              -xc.^3 .+ u
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

    @objective(model, Min, dot(w1, 0.5*(xc.^2 .+ u.^2)))

    #
    # Solve the problem
    #

    solve_and_print_solution(model)

    #
    # determine absolute and relative errors at collocation points
    #

    tau_err, ptau_err, xtau_err, w_err, D_err, A_err, K_err, P_err = psmethod(method, N+1)

    D1_err = 2.0 / (tf - ti) * D_err
    A1_err = (tf - ti) / 2.0 * A_err
    w1_err = w_err * (tf - ti) ./ 2.0

    L = lagrange_basis(ptau, tau_err)

    xc_err = lagrange_interpolation(L, value.(xp))

    L = lagrange_basis(tau, tau_err)

    u_err = lagrange_interpolation(L, value.(u))

    F_err = hcat(
              -xc_err.^3 .+ u_err
             )

    xc_err2 = value.(xi' .+ A1_err * F_err)

    abs_err = abs.(xc_err2 .- xc_err)

    rel_err = abs_err ./ (1.0.+maximum(abs.(xc_err), dims=1))

    max_rel_err = maximum(rel_err)

    @printf "maximum relative error: %e\n" max_rel_err

    p1 = Plots.plot(
                    xtau,
                    value.(x),
                    xlabel = "Tau",
                    ylabel = "State",
                    legend = false
                   )

    p2 = Plots.plot(
                    tau,
                    value.(u),
                    xlabel = "Tau",
                    ylabel = "Control",
                    legend = false
                   )

    display(Plots.plot(p1, p2, layout=(2,2), legend=false))
    readline()

    @assert is_solved_and_feasible(model)
end

hypersensitive()

