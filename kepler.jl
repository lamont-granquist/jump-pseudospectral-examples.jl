using JuMP
using LinearAlgebra
import Ipopt
using FastGaussQuadrature
using LegendrePolynomials

const r0 = [ 1 0 0 ]
const v0 = [ 0 1 0 ]

const n = 40

const t0 = 0
const tf = pi

const h = (tf - t0) / (n-1)

# second order finite difference differentiation matrix for n segments
function diffmat(n, h)
  # create the diagonals
  superdiagonal = fill(0.5/h,n)
  subdiagonal = fill(-0.5/h,n)
  D = diagm(-1=>subdiagonal,1=>superdiagonal)

  # Fix first and last rows.
  D[1,1:2] = [-1,1]/h
  D[n+1,n:n+1] = [-1,1]/h

  return D
end

# fourth order finite difference differentiation matrix for n segments
function diffmat2(n, h)
  # create the diagonals
  superdiagonal2 = fill(-1.0/12.0/h,n-1)
  superdiagonal = fill(2.0/3.0/h,n)
  subdiagonal = fill(-2.0/3.0/h,n)
  subdiagonal2 = fill(1.0/12.0/h,n-1)
  D = diagm(-2=>subdiagonal2,-1=>subdiagonal,1=>superdiagonal,2=>superdiagonal2)

  # Fix first and last rows.
  D[1,1:3] = [-1.5,2,-0.5]/h
  D[2,1:4] = [0,-1.5,2,-0.5]/h
  D[n,n-2:n+1] = [0.5,-2.0,1.5,0]/h
  D[n+1,n-1:n+1] = [0.5,-2.0,1.5]/h

  return D
end

# LGL PS differentiation matrix for n points
function diffmat3(n)
  x, w = gausslobatto(n)
  D = zeros(n, n)

  lp = Pl.(x, n-1)
  for i in 1:n
    for j in 1:n
      if i == j == 1
        D[i,j] = -n * (n - 1) / 4;
      elseif i == n && j == n
        D[i,j] = n * (n - 1) / 4;
      elseif i != j
        D[i,j] = lp[i] / lp[j] / (x[i] - x[j])
      else
        D[i,j] = 0;
      end
    end
  end

  return D
end

# LGR PS differentiation matrix for n points
function diffmat4(n)
  x, w = gaussradau(n)
  D = zeros(n, n)
  lp = Pl.(x, n-1)
  for i in 1:n
    for j in 1:n
      if i != j
        D[i,j] = lp[i] / lp[j] * (1.0 - x[j]) / ((1.0 - x[i]) * (x[i] - x[j]))
      elseif i == j == 1
        D[i,j] = -(n - 1) * (n + 1) / 4.0
      else
        D[i,j] = 1.0 / (2.0 * (1.0 - x[i]))
      end
    end
  end
  return D
end

# fLGR PS differentiation matrix for n points
function diffmat5(n)
  x, w = gaussradau(n)
  x = reverse(-x)
  w = reverse(w)
  D = zeros(n,n)

  # FIXME: does not work
  lp = Pl.(x, n-1)
  for i in 1:n
    for j in 1:n
      if i != j
        D[i,j] = lp[j] / lp[i] * (1.0 - x[i]) / ((1.0 - x[j]) * (x[j] - x[i]))
      elseif i == j == n
        D[i,j] = -(n - 1) * (n + 1) / 4.0
      else
        D[i,j] = 1.0 / (2.0 * (1.0 - x[j]))
      end
    end
  end
  return D
end

# LG PS differentiation matrix for n points
function diffmat6(n)
  x, w = gausslegendre(n)
  D = zeros(n, n)
  lp = dnPl.(x, n, 1)
  for i in 1:n
    for j in 1:n
      if i == j 
        D[i,j] = x[i] / (1 - x[i]^2)
      else
        D[i,j] = lp[i] / (lp[j] * (x[i] - x[j]))
      end
    end
  end
  return D
end

function kepler()
  model = Model(Ipopt.Optimizer)
  #set_attribute(model, "print_level", 8)

  @variables(model, begin
               r[1:n,1:3]
               v[1:n,1:3]
             end);

  fix(r[1,1], r0[1], force=true)
  fix(r[1,2], r0[2], force=true)
  fix(r[1,3], r0[3], force=true)
  fix(v[1,1], v0[1], force=true)
  fix(v[1,2], v0[2], force=true)
  fix(v[1,3], v0[3], force=true)

  for i in 1:n
    set_start_value(r[i,1], r0[1])
    set_start_value(r[i,2], r0[2])
    set_start_value(r[i,3], r0[3])
    set_start_value(v[i,1], v0[1])
    set_start_value(v[i,2], v0[2])
    set_start_value(v[i,3], v0[3])
  end

  #D = diffmat(n-1, h[1])
  #D = diffmat2(n-1, h[1])
  D = 2.0 / (tf - t0) * diffmat4(n)

  @expression(model, rm2[j = 1:n], sum(ri^2 for ri in r[j,:]))
  @expression(model, rm[j = 1:n], sqrt(rm2[j]))
  @expression(model, rm3[j = 1:n], rm2[j]*rm[j])

  @constraint(model, D * r == v)
  @constraint(model, D * v == -r ./ rm3)

  optimize!(model)

  display(value.(r[n,:]))
  display(value.(v[n,:]))

  @assert is_solved_and_feasible(model)
end

kepler()
