using JuMP

# This is copied from the GPOPS-II user manual, translated to Julia and then
# modified to work with JuMP

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
    i = acos(hv[3]/sqrt(h2))
    Om = acos(nv[1]/n)
    Om = op_ifelse(
              op_strictly_less_than( nv[2], 0-eps()),
              2*pi-Om,
              Om
             )
    #if nv[2]<0-eps()
    #    Om = 2*pi-Om
    #end
    om = acos(nv'*ev/n/e)
    om = op_ifelse(
                   op_strictly_less_than( ev[3], 0),
                   2*pi-om,
                   om
                  )
    #if ev[3]<0
    #    om = 2*pi-om
    #end
    nu = acos(ev'*rv/e/r)
    nu = op_ifelse(
                    op_strictly_less_than(rv'*vv, 0),
                    2*pi-nu,
                    nu
                   )
    #if rv'*vv<0
    #    nu = 2*pi-nu
    #end
    return [a e i Om om]
end
