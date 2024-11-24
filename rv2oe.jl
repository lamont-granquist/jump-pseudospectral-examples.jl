using JuMP

# This is copied from the GPOPS-II user manual, translated to Julia and then
# modified to work with JuMP

function safe_acos(x)
    x = op_ifelse(op_strictly_less_than(x, -1), -1, x)
    x = op_ifelse(op_strictly_greater_than(x, 1), 1, x)
    return acos(x)
end

function rv2oe(mu, rv, vv)
    K = [0;0;1]
    hv = cross(rv,vv)
    nv = cross(K,hv)
    n = sqrt(nv'*nv)
    h2 = (hv'*hv)
    v2 = (vv'*vv)
    r = sqrt(rv'*rv)
    ev = 1/mu *( (v2-mu/r)*rv - (rv'*vv)*vv )
    p = h2/mu
    e = sqrt(ev'*ev)
    a = p/(1-e*e)
    i = safe_acos(hv[3]/sqrt(h2))
    Om = safe_acos(nv[1]/n)
    Om = op_ifelse(
              op_strictly_less_than( nv[2], 0-eps()),
              2*pi-Om,
              Om
             )
    om = safe_acos(nv'*ev/n/e)
    om = op_ifelse(
                   op_strictly_less_than( ev[3], 0),
                   2*pi-om,
                   om
                  )
    nu = safe_acos(ev'*rv/e/r)
    nu = op_ifelse(
                    op_strictly_less_than(rv'*vv, 0),
                    2*pi-nu,
                    nu
                   )
    return [a e i Om om nu]
end
