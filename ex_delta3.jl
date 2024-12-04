
# Benson, D. (2005). A Gauss pseudospectral transcription for optimal control
# (Doctoral dissertation, Massachusetts Institute of Technology).
#
# Betts, J. T. (2010). Practical methods for optimal control and estimation using
# nonlinear programming. Society for Industrial and Applied Mathematics.
#
# Patterson, M. A., & Rao, A. V. (2014). GPOPS-II: A MATLAB software for solving
# multiple-phase optimal control problems using hp-adaptive Gaussian quadrature
# collocation methods and sparse nonlinear programming. ACM Transactions on
# Mathematical Software (TOMS), 41(1), 1-37.

# TODO:
#  - integral formulation for LG/LGL
#  - costate estimation for LG/LGL/LGR
#  - validate costate with indirect shooting method
#  - endpoint control estimation - Huntington(2008)
#  - chebyshev at gauss and radau points?
#    - chebyshev-lobatto costate - Gong(2010)
#    - chebyshev-gauss - Tang(2015)
#    - hermite-lobatto - Liu(2014)
#    - better legendre-lobatto -Garrido(2023)
#  - generate control law - Tian(2011)
#  - mesh refinement - Liu(2015)
#  - convexification
#  - birkhoff PS methods?

using JuMP
import Ipopt
using SatelliteToolboxBase
import Plots
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using Glob

foreach(include, glob("*.jl", "lib"))

# CGL, LGL, LGR, LG
const method = "LGR"

# supported by LGR and LG methods
const integral = false

# supported by LGR differentiation method
const costate = true

# number of grid points
const N = 10

#
# Earth
#

const µ🜨   = 3.986012e14   # m³/s²
const r🜨   = 6378145       # m
const ω🜨   = 7.29211585e-5 # rad/s
const rho0 = 1.225         # kg/m³
const H0   = 7200          # m

#
# initial conditions
#

const lat = 28.5 # °
const lng = 0    # °

#
# terminal conditions
#

const smaf  = 24361140
const eccf  = 0.7308
const incf  = deg2rad(28.5)
const lanf  = deg2rad(269.8)
const argpf = deg2rad(130.5)

#
# vehicle constants
#

const Aref           = 4 * pi  # m²
const Cd             = 0.5

const srbWetMass     = 19290  # kg
const srbPropMass    = 17010 # kg
const srbBurnTime    = 75.2  # sec
const srbThrust      = 628500  # N
const srbMdot        = srbPropMass / srbBurnTime

const firstWetMass   = 104380 # kg
const firstPropMass  = 95550 # kg
const firstBurnTime  = 261   # sec
const firstThrust    = 1083100 # N
const firstMdot      = firstPropMass / firstBurnTime

const secondWetMass  = 19300  # kg
const secondPropMass = 16820 # kg
const secondBurnTime = 700   # sec
const secondThrust   = 110094  # N
const secondMdot     = secondPropMass / secondBurnTime

const payloadMass    = 4164 # kg

#
# derived constants
#

const Ω = ω🜨 * [0 -1 0; 1 0 0; 0 0 0]
const lati = lat*π/180
const lngi = lng*π/180
const r1i = r🜨 * [ cos(lati)*cos(lngi),cos(lati)*sin(lngi),sin(lati) ]
const v1i = Ω * r1i
const m1i = payloadMass + secondWetMass + firstWetMass + 9 * srbWetMass
const m1f = m1i - (firstMdot+6*srbMdot) * srbBurnTime
const m2i = m1i - firstMdot * srbBurnTime - 6 * srbWetMass
const m2f = m2i - (firstMdot+3*srbMdot) * srbBurnTime
const m3i = m1i - 2*firstMdot * srbBurnTime - 9 * srbWetMass
const m3f = m1i - 9 * srbWetMass - firstMdot * firstBurnTime
const m4i = payloadMass + secondWetMass
const m4f = payloadMass
const mdot1 = firstMdot + 6 * srbMdot
const mdot2 = firstMdot + 3 * srbMdot
const mdot3 = firstMdot
const mdot4 = secondMdot
const dt1 = srbBurnTime
const dt2 = srbBurnTime
const dt3 = firstBurnTime - srbBurnTime * 2
const dt4 = secondBurnTime
const T1 = firstThrust + 6 * srbThrust
const T2 = firstThrust + 3 * srbThrust
const T3 = firstThrust
const T4 = secondThrust
const rmax = 2*r🜨
const vmax = 10000
const umax = 10
const rmin = -rmax
const vmin = -vmax
const umin = -umax

#
# scaling
#

const r_scale = norm(r1i)
const v_scale = sqrt(µ🜨/r_scale)
const t_scale = r_scale / v_scale
const m_scale = m1i
const a_scale = v_scale / t_scale
const f_scale = m_scale * a_scale
const area_scale = r_scale^2
const vol_scale = r_scale * area_scale
const d_scale = m_scale / vol_scale
const mdot_scale = m_scale / t_scale

#
# applying scaling
#

const r1is = r1i / r_scale
const v1is = v1i / v_scale
const v1isnorm = norm(v1is)
const m1is = m1i / m_scale
const m2is = m2i / m_scale
const m3is = m3i / m_scale
const m4is = m4i / m_scale
const m1fs = m1f / m_scale
const m2fs = m2f / m_scale
const m3fs = m3f / m_scale
const m4fs = m4f / m_scale
const mdot1s = mdot1 / mdot_scale
const mdot2s = mdot2 / mdot_scale
const mdot3s = mdot3 / mdot_scale
const mdot4s = mdot4 / mdot_scale
const dt1s = dt1 / t_scale
const dt2s = dt2 / t_scale
const dt3s = dt3 / t_scale
const dt4s = dt4 / t_scale
const T1s = T1 / f_scale
const T2s = T2 / f_scale
const T3s = T3 / f_scale
const T4s = T4 / f_scale
const rmins = rmin / r_scale
const rmaxs = rmax / r_scale
const vmins = vmin / v_scale
const vmaxs = vmax / v_scale
const r🜨s = r🜨 / r_scale
const rho0s = rho0 / d_scale
const H0s = H0 / r_scale
const Arefs = Aref / area_scale
const Ωs = Ω * t_scale
const smafs = smaf / r_scale

#
# better terminal conditions
#

oe = KeplerianElements(0, smaf, eccf, incf, lanf, argpf, 0)
rf, vf = kepler_to_rv(oe)
rf = rf / r_scale
vf = vf / v_scale
hf = cross(rf, vf)
ef = cross(vf, hf) - rf / norm(rf)

function delta3()
  model = Model(Ipopt.Optimizer)
  set_optimizer_attribute(model, "print_level", 5)
  set_optimizer_attribute(model, "print_user_options", "yes")
  set_optimizer_attribute(model, "max_iter", 2000)
  set_optimizer_attribute(model, "tol", 1e-12)
  #set_optimizer_attribute(model, "mumps_permuting_scaling", 7)
  #set_optimizer_attribute(model, "mumps_scaling", 8)
  #set_optimizer_attribute(model, "nlp_scaling_method", "none")

  tau, ptau, xtau, w, D, A, K, P = psmethod(method, N)

  #
  # Initial Guess Generation
  #
  # [ Based on Benson(2005) but launch with fixed inertial heading 45 degrees up and due east ]
  #

  # inertial heading
  elev = deg2rad(45)
  az = deg2rad(90)
  u_enu = [ cos(elev)*sin(az); cos(elev)*cos(az); sin(elev) ]
  R_toecef = [
              -sin(lngi) -sin(lati)*cos(lngi) cos(lati)*cos(lngi);
              cos(lngi)  -sin(lati)*sin(lngi) cos(lati)*sin(lngi);
              0          cos(lati)            sin(lati)
             ];
  u_ecef = R_toecef * u_enu

  # simplified vacuum rocket model
  function rocket_stage!(dx, x, p, t)
    u = p[1:3]; T = p[4]; mdot = p[5]; dt = p[6]
    r = x[1:3]; v = x[4:6]; m = x[7]

    r_norm = norm(r)
    dx[1:3] = v
    dx[4:6] = -r/r_norm^3 + T/m * u
    dx[7] = -mdot
    dx .= dx * dt / 2
  end

  # stage 1
  x0 = [ r1is; v1is; m1is ]
  p = [ u_ecef; T1s; mdot1s; dt1s ]

  prob = ODEProblem(rocket_stage!, x0, (-1.0, 1.0), p)
  sol = solve(prob, Tsit5(), saveat=xtau)

  r1init = hcat(sol(xtau)...)[1:3, :]
  v1init = hcat(sol(xtau)...)[4:6, :]
  m1init = hcat(sol(xtau)...)[7, :]

  # stage 2
  x0 = [ r1init[:,end]; v1init[:,end]; m2is ]
  p = [ u_ecef; T2s; mdot2s; dt2s ]

  prob = ODEProblem(rocket_stage!, x0, (-1.0, 1.0), p)
  sol = solve(prob, Tsit5(), saveat=xtau)

  r2init = hcat(sol(xtau)...)[1:3, :]
  v2init = hcat(sol(xtau)...)[4:6, :]
  m2init = hcat(sol(xtau)...)[7, :]

  # stage 3
  x0 = [ r2init[:,end]; v2init[:,end]; m3is ]
  p = [ u_ecef; T3s; mdot3s; dt3s ]

  prob = ODEProblem(rocket_stage!, x0, (-1.0, 1.0), p)
  sol = solve(prob, Tsit5(), saveat=xtau)

  r3init = hcat(sol(xtau)...)[1:3, :]
  v3init = hcat(sol(xtau)...)[4:6, :]
  m3init = hcat(sol(xtau)...)[7, :]

  # stage 4

  dt4guess = dt4s
  x0 = [ r3init[:,end]; v3init[:,end]; m4is ]
  p = [ u_ecef; T4s; mdot4s; dt4guess ]

  prob = ODEProblem(rocket_stage!, x0, (-1.0, 1.0), p)
  sol = solve(prob, Tsit5(), saveat=xtau)

  r4init = hcat(sol(xtau)...)[1:3, :]
  v4init = hcat(sol(xtau)...)[4:6, :]
  m4init = hcat(sol(xtau)...)[7, :]

  #
  # Variables
  #

  @variable(model, rmins <= r1[i=1:N,j=1:3] <= rmaxs, start=r1init[j,i])
  @variable(model, vmins <= v1[i=1:N,j=1:3] <= vmaxs, start=v1init[j,i])
  @variable(model, m1fs <= m1[i=1:N] <= m1is, start=m1init[i])
  @variable(model, umin <= u1[i=1:K,j=1:3] <= umax, start=u_ecef[j])
  @variable(model, ti1, start=0)
  @variable(model, tf1, start=dt1s)

  @variable(model, rmins <= r2[i=1:N,j=1:3] <= rmaxs, start=r2init[j,i])
  @variable(model, vmins <= v2[i=1:N,j=1:3] <= vmaxs, start=v2init[j,i])
  @variable(model, m2fs <= m2[i=1:N] <= m2is, start=m2init[i])
  @variable(model, umin <= u2[i=1:K,j=1:3] <= umax, start=u_ecef[j])
  @variable(model, ti2, start=dt1s)
  @variable(model, tf2, start=dt1s+dt2s)

  @variable(model, rmins <= r3[i=1:N,j=1:3] <= rmaxs, start=r3init[j,i])
  @variable(model, vmins <= v3[i=1:N,j=1:3] <= vmaxs, start=v3init[j,i])
  @variable(model, m3fs <= m3[i=1:N] <= m3is, start=m3init[i])
  @variable(model, umin <= u3[i=1:K,j=1:3] <= umax, start=u_ecef[j])
  @variable(model, ti3, start=dt1s+dt2s)
  @variable(model, tf3, start=dt1s+dt2s+dt3s)

  @variable(model, rmins <= r4[i=1:N,j=1:3] <= rmaxs, start=r4init[j,i])
  @variable(model, vmins <= v4[i=1:N,j=1:3] <= vmaxs, start=v4init[j,i])
  @variable(model, m4fs <= m4[i=1:N] <= m4is, start=m4init[i])
  @variable(model, umin <= u4[i=1:K,j=1:3] <= umax, start=u_ecef[j])
  @variable(model, ti4, start=dt1s+dt2s+dt3s)
  @variable(model, tf4, start=dt1s+dt2s+dt3s+dt4guess)

  #
  # Endpoint variable slices
  #

  x1i = vcat(r1[1,:], v1[1,:], m1[1])
  x2i = vcat(r2[1,:], v2[1,:], m2[1])
  x3i = vcat(r3[1,:], v3[1,:], m3[1])
  x4i = vcat(r4[1,:], v4[1,:], m4[1])

  x1f = vcat(r1[N,:], v1[N,:], m1[N])
  x2f = vcat(r2[N,:], v2[N,:], m2[N])
  x3f = vcat(r3[N,:], v3[N,:], m3[N])
  x4f = vcat(r4[N,:], v4[N,:], m4[N])

  #
  # Collocated variable slices
  #

  offset = 0
  if method == "LGR" || method == "LG"
      offset = 1
  end

  @expression(model, r1c[i=1:K,j=1:3], r1[i+offset,j])
  @expression(model, v1c[i=1:K,j=1:3], v1[i+offset,j])
  @expression(model, m1c[i=1:K], m1[i+offset])
  x1c = hcat(r1c, v1c, m1c)

  @expression(model, r2c[i=1:K,j=1:3], r2[i+offset,j])
  @expression(model, v2c[i=1:K,j=1:3], v2[i+offset,j])
  @expression(model, m2c[i=1:K], m2[i+offset])
  x2c = hcat(r2c, v2c, m2c)

  @expression(model, r3c[i=1:K,j=1:3], r3[i+offset,j])
  @expression(model, v3c[i=1:K,j=1:3], v3[i+offset,j])
  @expression(model, m3c[i=1:K], m3[i+offset])
  x3c = hcat(r3c, v3c, m3c)

  @expression(model, r4c[i=1:K,j=1:3], r4[i+offset,j])
  @expression(model, v4c[i=1:K,j=1:3], v4[i+offset,j])
  @expression(model, m4c[i=1:K], m4[i+offset])
  x4c = hcat(r4c, v4c, m4c)

  #
  # Polynomial variable slices (FIXME: is this the right term?)
  #

  @expression(model, r1p[i=1:P,j=1:3], r1[i,j])
  @expression(model, v1p[i=1:P,j=1:3], v1[i,j])
  @expression(model, m1p[i=1:P], m1[i])
  x1p = hcat(r1p, v1p, m1p)

  @expression(model, r2p[i=1:P,j=1:3], r2[i,j])
  @expression(model, v2p[i=1:P,j=1:3], v2[i,j])
  @expression(model, m2p[i=1:P], m2[i])
  x2p = hcat(r2p, v2p, m2p)

  @expression(model, r3p[i=1:P,j=1:3], r3[i,j])
  @expression(model, v3p[i=1:P,j=1:3], v3[i,j])
  @expression(model, m3p[i=1:P], m3[i])
  x3p = hcat(r3p, v3p, m3p)

  @expression(model, r4p[i=1:P,j=1:3], r4[i,j])
  @expression(model, v4p[i=1:P,j=1:3], v4[i,j])
  @expression(model, m4p[i=1:P], m4[i])
  x4p = hcat(r4p, v4p, m4p)

  #
  # Fixed constraints
  #

  fix.(r1[1,:], r1is; force = true)
  fix.(v1[1,:], v1is; force = true)
  fix(m1[1], m1is, force=true)
  fix(m2[1], m2is, force=true)
  fix(m3[1], m3is, force=true)
  fix(m4[1], m4is, force=true)
  fix(ti1, 0, force=true)
  fix(tf1, dt1s, force=true)
  fix(ti2, dt1s, force=true)
  fix(tf2, dt1s+dt2s, force=true)
  fix(ti3, dt1s+dt2s, force=true)
  fix(tf3, dt1s+dt2s+dt3s, force=true)
  fix(ti4, dt1s+dt2s+dt3s, force=true)

  #
  # Dynamical constraints
  #

  D1 = 2.0 / (tf1 - ti1) * D
  D2 = 2.0 / (tf2 - ti2) * D
  D3 = 2.0 / (tf3 - ti3) * D
  D4 = 2.0 / (tf4 - ti4) * D

  A1 = (tf1 - ti1) / 2.0 * A
  A2 = (tf2 - ti2) / 2.0 * A
  A3 = (tf3 - ti3) / 2.0 * A
  A4 = (tf4 - ti4) / 2.0 * A

  w1 = w * (tf1 - ti1) ./ 2.0
  w2 = w * (tf2 - ti2) ./ 2.0
  w3 = w * (tf3 - ti3) ./ 2.0
  w4 = w * (tf4 - ti4) ./ 2.0

  # Stage 1

  @expression(model, r1ccube[i=1:K], sqrt(sum(r^2 for r in r1c[i,:]))^3)
  @expression(model, r1cnorm[i=1:K], sqrt(sum(r^2 for r in r1c[i,:])))
  @expression(model, v1rel, v1c .- r1c * Ωs')
  @expression(model, v1relnorm[i=1:K], sqrt(1e-8 + sum(v^2 for v in v1rel[i,:])))
  @expression(model, rho1[i=1:K], rho0s*exp(-(r1cnorm[i] - r🜨s)/H0s))
  @expression(model, Drag1[i=1:K,j=1:3], -0.5*Cd*Arefs*rho1[i]*v1relnorm[i]*v1rel[i,j])

  F1 = hcat(
            v1c,
            -r1c ./ r1ccube + T1s * u1 ./ m1c + Drag1 ./ m1c,
            -mdot1s * ones(K),
           )

  if integral
      @constraint(model, x1c == x1i' .+ A1 * F1)

      if method == "LG"
          @constraint(model, x1f == x1i + F1' * w1)
      end
  else
      @constraint(model, dyn1, D1 * x1p == F1)
      if method == "LG"
        @constraint(model, x1f == x1i + F1' * w1)
      end
  end

  # Stage 2

  @expression(model, rcube2[i = 1:K], sqrt(sum(r^2 for r in r2c[i,:]))^3)
  @expression(model, r2cnorm[i = 1:K], sqrt(sum(r^2 for r in r2c[i,:])))
  @expression(model, v2rel, v2c .- r2c * Ωs')
  @expression(model, v2relnorm[i=1:K], sqrt(1e-8 + sum(v^2 for v in v2rel[i,:])))
  @expression(model, rho2[i=1:K], rho0s*exp(-(r2cnorm[i] - r🜨s)/H0s))
  @expression(model, Drag2[i=1:K,j=1:3], -0.5*Cd*Arefs*rho2[i]*v2relnorm[i]*v2rel[i,j])

  F2 = hcat(
            v2c,
            -r2c ./ rcube2 + T2s * u2 ./ m2c + Drag2 ./ m2c,
            -mdot2s * ones(K),
           )

  if integral
      @constraint(model, x2c == x2i' .+ A2 * F2)

      if method == "LG"
          @constraint(model, x2f == x2i + F2' * w2)
      end
  else
      @constraint(model, dyn2, D2 * x2p == F2)
      if method == "LG"
          @constraint(model, x2f == x2i + F2' * w2)
      end
  end


  # Stage 3

  @expression(model, rcube3[i = 1:K], sqrt(sum(r^2 for r in r3c[i,:]))^3)
  @expression(model, r3cnorm[i = 1:K], sqrt(sum(r^2 for r in r3c[i,:])))
  @expression(model, v3rel, v3c .- r3c * Ωs')
  @expression(model, v3relnorm[i=1:K], sqrt(1e-8 + sum(v^2 for v in v3rel[i,:])))
  @expression(model, rho3[i=1:K], rho0s*exp(-(r3cnorm[i] - r🜨s)/H0s))
  @expression(model, Drag3[i=1:K,j=1:3], -0.5*Cd*Arefs*rho3[i]*v3relnorm[i]*v3rel[i,j])

  F3 = hcat(
            v3c,
            -r3c ./ rcube3 + T3s * u3 ./ m3c + Drag3 ./ m3c,
            -mdot3s * ones(K),
           )

  if integral
      @constraint(model, x3c == x3i' .+ A3 * F3)

      if method == "LG"
          @constraint(model, x3f == x3i + F3' * w3)
      end
  else
      @constraint(model, dyn3, D3 * x3p == F3)

      if method == "LG"
          @constraint(model, x3f == x3i + F3' * w3)
      end
  end

  # Stage 4

  @expression(model, rcube4[i = 1:K], sqrt(sum(r^2 for r in r4c[i,:]))^3)
  @expression(model, r4cnorm[i = 1:K], sqrt(sum(r^2 for r in r4c[i,:])))
  @expression(model, v4rel, v4c .- r4c * Ωs')
  @expression(model, v4relnorm[i=1:K], sqrt(1e-8 + sum(v^2 for v in v4rel[i,:])))
  @expression(model, rho4[i=1:K], rho0s*exp(-(r4cnorm[i] - r🜨s)/H0s))
  @expression(model, Drag4[i=1:K,j=1:3], -0.5*Cd*Arefs*rho4[i]*v4relnorm[i]*v4rel[i,j])

  F4 = hcat(
            v4c,
            -r4c ./ rcube4 + T4s * u4 ./ m4c + Drag4 ./ m4c,
            -mdot4s * ones(K),
           )

  if integral
      @constraint(model, x4c == x4i' .+ A4 * F4)

      if method == "LG"
          @constraint(model, x4f == x4i + F4' * w4)
      end
  else
      @constraint(model, dyn4, D4 * x4p == F4)

      if method == "LG"
          @constraint(model, x4f == x4i + F4' * w4)
      end
  end

  #
  # Path constraints
  #

  @expression(model, r1norm[i = 1:N], sum(r^2 for r in r1[i,:]))
  @constraint(model, r1norm >= ones(N))

  @expression(model, r2norm[i = 1:N], sum(r^2 for r in r2[i,:]))
  @constraint(model, r2norm >= ones(N))

  @expression(model, r3norm[i = 1:N], sum(r^2 for r in r3[i,:]))
  @constraint(model, r3norm >= ones(N))

  @expression(model, r4norm[i = 1:N], sum(r^2 for r in r4[i,:]))
  @constraint(model, r4norm >= ones(N))

  #
  # Continuity constraints
  #

  @constraint(model, r1[N,:] == r2[1,:])
  @constraint(model, v1[N,:] == v2[1,:])
  @constraint(model, tf1 == ti2)

  @constraint(model, r2[N,:] == r3[1,:])
  @constraint(model, v2[N,:] == v3[1,:])
  @constraint(model, tf2 == ti3)

  @constraint(model, r3[N,:] == r4[1,:])
  @constraint(model, v3[N,:] == v4[1,:])
  @constraint(model, tf3 == ti4)

  #
  # Control constraints
  #

  @expression(model, u1norm[i = 1:K], sum(u^2 for u in u1[i,:]))
  @expression(model, u2norm[i = 1:K], sum(u^2 for u in u2[i,:]))
  @expression(model, u3norm[i = 1:K], sum(u^2 for u in u3[i,:]))
  @expression(model, u4norm[i = 1:K], sum(u^2 for u in u4[i,:]))

  @constraint(model, u1norm <= ones(K))
  @constraint(model, u2norm <= ones(K))
  @constraint(model, u3norm <= ones(K))
  @constraint(model, u4norm <= ones(K))

  #
  # Time constraints
  #

  @constraint(model, tf1 - ti1 == dt1s)
  @constraint(model, tf2 - ti2 == dt2s)
  @constraint(model, tf3 - ti3 == dt3s)
  @constraint(model, tf4 >= ti4)

  #
  # Terminal constraints
  #

  @expression(model, h4f, cross(r4[N,1:3], v4[N,1:3]))
  @constraint(model, h4f == hf)
  @expression(model, r4fnorm, sqrt(sum(r^2 for r in r4[N,:])))
  @expression(model, e4f, cross(v4[N,1:3], h4f) - r4[N,1:3] ./ r4fnorm)
  @constraint(model, e4f == ef)

  #
  # Objective
  #

  @objective(model, Max, m4[N])

  #
  # Solve the problem
  #

  solve_and_print_solution(model)

  #
  # Display output
  #

  @printf "\n"
  @printf "first stage burntime:  %6.2f s\n" value(tf1 - ti1) * t_scale
  @printf "second stage burntime: %6.2f s\n" value(tf2 - ti2) * t_scale
  @printf "third stage burntime:  %6.2f s\n" value(tf3 - ti3) * t_scale
  @printf "fourth stage burntime: %6.2f s\n" value(tf4 - ti4) * t_scale
  @printf "\n"

  tbt = value(tf4 - ti1) * t_scale
  mf = value(m4[N]) * m_scale

  tbt_betts = 924.139
  mf_betts = 7529.712412

  @printf "total burntime: %.2f s (acc: %e)\n" tbt (tbt - tbt_betts)/tbt_betts
  @printf "delivered mass: %.2f kg (acc: %e)\n" mf (mf - mf_betts)/mf_betts
  @printf "\n"

  rf = value.(r4[N,:]) * r_scale
  vf = value.(v4[N,:]) * v_scale

  oe = rv2oe(µ🜨, rf, vf)

  sma = oe[1]
  ecc = oe[2]
  inc = oe[3]
  lan = oe[4]
  argp = oe[5]
  nu = oe[6]

  @printf "sma:  %.2f km\n" sma
  @printf "ecc:  %4f\n" ecc
  @printf "inc:  %6.2f°\n" rad2deg(inc)
  @printf "lan:  %6.2f°\n" rad2deg(lan)
  @printf "argp: %6.2f°\n" rad2deg(argp)
  @printf "nu:   %6.2f°\n" rad2deg(nu)

  #
  # Descale and interpolate the variables
  #

  range = LinRange(-1,1,20)
  L = lagrange_basis(ptau, range)

  r1 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(r1p))) * r_scale
  r2 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(r2p))) * r_scale
  r3 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(r3p))) * r_scale
  r4 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(r4p))) * r_scale

  v1 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(v1p))) * v_scale
  v2 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(v2p))) * v_scale
  v3 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(v3p))) * v_scale
  v4 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(v4p))) * v_scale

  m1 = lagrange_interpolation(L, value.(m1p)) * m_scale
  m2 = lagrange_interpolation(L, value.(m2p)) * m_scale
  m3 = lagrange_interpolation(L, value.(m3p)) * m_scale
  m4 = lagrange_interpolation(L, value.(m4p)) * m_scale

  L = lagrange_basis(tau, range)

  u1 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(u1)))
  u2 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(u2)))
  u3 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(u3)))
  u4 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(u4)))

  #
  # Construct ranges of real times at the interpolation points
  #

  ti1 = value(ti1)
  tf1 = value(tf1)
  ti2 = value(ti2)
  tf2 = value(tf2)
  ti3 = value(ti3)
  tf3 = value(tf3)
  ti4 = value(ti4)
  tf4 = value(tf4)

  t1 = (range * (tf1 - ti1) ./ 2.0 .+ (tf1 + ti1) / 2.0 ) * t_scale
  t2 = (range * (tf2 - ti2) ./ 2.0 .+ (tf2 + ti2) / 2.0 ) * t_scale
  t3 = (range * (tf3 - ti3) ./ 2.0 .+ (tf3 + ti3) / 2.0 ) * t_scale
  t4 = (range * (tf4 - ti4) ./ 2.0 .+ (tf4 + ti4) / 2.0 ) * t_scale

  #
  # Combine the phases and determine the norms of the 3-vectors
  #

  r = [r1; r2; r3; r4]
  v = [v1; v2; v3; v4]
  m = [m1; m2; m3; m4]
  t = [t1; t2; t3; t4]
  u = [u1; u2; u3; u4]

  rnorm = norm.(eachrow(r))
  vnorm = norm.(eachrow(v))
  unorm = norm.(eachrow(u))

  #
  # Pull costate estimate out of the KKT multipliers and interpolate
  #

  if !integral && costate
      D0 = D[:, 1]

      Λ1 = dual(dyn1)
      Λ2 = dual(dyn2)
      Λ3 = dual(dyn3)
      Λ4 = dual(dyn4)

      λ1 = vcat(
                -D0' * Λ1,
                Λ1 ./ w,
               )
      λ2 = vcat(
                -D0' * Λ2,
                Λ2 ./ w,
               )
      λ3 = vcat(
                -D0' * Λ3,
                Λ3 ./ w,
               )
      λ4 = vcat(
                -D0' * Λ4,
                Λ4 ./ w,
               )

      range = LinRange(-1,1,20)
      L = lagrange_basis(ptau, range)

      λ1 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(λ1)))
      λ2 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(λ2)))
      λ3 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(λ3)))
      λ4 = reduce(hcat,lagrange_interpolation(L, col) for col in eachcol(value.(λ4)))

      λ = [λ1; λ2; λ3; λ4]

      # not entirely sure where the minus sign comes from here
      λr = -λ[:,1:3]
      λv = -λ[:,4:6]
      λm = -λ[:,7]

      λvnorm = norm.(eachrow(λv))
      λvunit = λv ./ λvnorm
  end

  #
  # Do some plotting of interpolated results
  #

  p1 = Plots.plot(
                 t,
                 m ./ 1000,
                 xlabel = "Time (s)",
                 ylabel = "Mass (t)",
                 legend = false
                )
  p2 = Plots.plot(
                 t,
                 (rnorm .- r🜨) ./ 1000,
                 xlabel = "Time (s)",
                 ylabel = "Height (km)",
                 legend = false
                )
  p3 = Plots.plot(
                 t,
                 vnorm ./ 1000,
                 xlabel = "Time (s)",
                 ylabel = "Velocity (km/s)",
                 legend = false
                )
  p4 = Plots.plot(
                 t,
                 [ u[:,1] u[:,2] u[:,3] unorm ],
                 xlabel = "Time (s)",
                 ylabel = "Control",
                 legend = false
                )
  p5 = Plots.plot(
                 t,
                 λvnorm,
                 xlabel = "Time (s)",
                 ylabel = "λv Magnitude",
                 legend = false
                 )
  p6 = Plots.plot(
                 t,
                 [ λvunit[:,1] λvunit[:,2] λvunit[:,3] ],
                 xlabel = "Time (s)",
                 ylabel = "λv Direction",
                 legend = false
                )

  display(Plots.plot(p1, p2, p3, p4, p5, p6, layout=(3,2), legend=false))
  readline()

  @assert is_solved_and_feasible(model)
end

delta3()
