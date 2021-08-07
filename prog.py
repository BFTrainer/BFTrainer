import numpy as np
import gurobipy as grb
from scipy import interpolate

def interp1d4s(Ns, Os, Nx):
    f = interpolate.interp1d(Ns, Os, bounds_error=False, fill_value="extrapolate")
    return float(f(Nx))

def re_allocate(cmap, jmin, jmax, Ns, Os, Tfwd, res_up, res_dw, time_limit): # tfwd 10 time_limit - 30 seconds
    nJ, nN = cmap.shape
    J = range(nJ)
    N = range(nN)
    assert len(jmin) == nJ and len(jmax) == nJ and len(res_up) == nJ and len(res_dw) == nJ
    # compuate throughput of current map, cost will depend on it
    _cNj = cmap.sum(axis=1)
    c_rate = [interp1d4s(Ns[_j], Os[_j], _cNj[_j]) \
              if _cNj[_j] >= jmin[_j] else 0 for _j in range(cmap.shape[0])]
    
    c_map = cmap.tolist()

    opt_mdl = grb.Model()
    opt_mdl.Params.OutputFlag = 0
    opt_mdl.Params.TimeLimit = time_limit
    
    x = opt_mdl.addVars(nJ, nN, vtype=grb.GRB.BINARY, name='x')
    bigM = nN + 1
    
    dummy4Jmin = opt_mdl.addVars(nJ, vtype=grb.GRB.BINARY, name='dummy4Jmin')
    dummy4Jmax = opt_mdl.addVars(nJ, vtype=grb.GRB.BINARY, name='dummy4Jmax')

    # job size constraints
    for _j in J:
        _nJ = grb.quicksum(x[_j, _n] for _n in N)

        opt_mdl.addConstr( _nJ >= jmin[_j] - bigM * dummy4Jmin[_j] )
        opt_mdl.addConstr( _nJ <= 0 + bigM * (1 - dummy4Jmin[_j]) )

        opt_mdl.addConstr( jmax[_j] >= _nJ - bigM * dummy4Jmax[_j] )
        opt_mdl.addConstr( _nJ <= 0 + bigM * (1 - dummy4Jmax[_j]) )
    
    # node allocations constraint
    for _n in N:
        opt_mdl.addConstr( grb.quicksum(x[_j, _n] for _j in J) <= 1 )# at most one application

    # constraint to disallow job migration 
    dummy4xxorc  = opt_mdl.addVars(nJ, nN, vtype=grb.GRB.BINARY) # x XOR c
    dummy4migr   = opt_mdl.addVars(nJ, vtype=grb.GRB.BINARY)
    c = c_map # a shorter name for easy ref
    for _j in J:
        for _n in N:
            opt_mdl.addConstr( dummy4xxorc[_j, _n] <= x[_j, _n] + c[_j][_n] )
            opt_mdl.addConstr( dummy4xxorc[_j, _n] >= x[_j, _n] - c[_j][_n] )
            opt_mdl.addConstr( dummy4xxorc[_j, _n] >= c[_j][_n] - x[_j, _n] )
            opt_mdl.addConstr( dummy4xxorc[_j, _n] <= 2 - x[_j, _n] - c[_j][_n] )

    for _j in J:
        _xxorc_sumJ = grb.quicksum(dummy4xxorc[_j, _n] for _n in N)
        _nj_new = grb.quicksum(x[_j, _n] for _n in N)
        _nj_old = sum(c[_j])
        opt_mdl.addConstr( _nj_new - _nj_old >= _xxorc_sumJ  - bigM * dummy4migr[_j] )
        opt_mdl.addConstr( _nj_new - _nj_old <= -_xxorc_sumJ + bigM * (1 - dummy4migr[_j]) )

    # approximate the scalability
    w4sappro = {_j:opt_mdl.addVars(len(Ns[_j]), lb=0.0, ub=1.0, vtype=grb.GRB.CONTINUOUS) for _j in J}

    S_approx = {}
    for _j in J:
        I = range(len(Ns[_j]))
        opt_mdl.addConstr( grb.quicksum(w4sappro[_j]) == 1 ) # convexification

        _Nj = grb.quicksum(x[_j, _n] for _n in N)
        opt_mdl.addConstr( _Nj == grb.quicksum(Ns[_j][_i] * w4sappro[_j][_i] for _i in I) )

        opt_mdl.addSOS(grb.GRB.SOS_TYPE2, w4sappro[_j], Ns[_j])

        S_approx[_j] = grb.quicksum(Os[_j][_i] * w4sappro[_j][_i] for _i in I)

    # scale up/down cost
    dummy4upcost = opt_mdl.addVars(nJ, vtype=grb.GRB.BINARY)
    dummy4dwcost = opt_mdl.addVars(nJ, vtype=grb.GRB.BINARY)

    for _j in J:
        _Nj = grb.quicksum(x[_j, _n] for _n in N)
        _Cj = sum(c_map[_j])
        opt_mdl.addConstr( _Nj <= _Cj + (bigM - _Cj) * dummy4upcost[_j] )
        opt_mdl.addConstr( _Nj >= (_Cj + 1) * dummy4upcost[_j] )

        opt_mdl.addConstr( _Nj <= (_Cj - 1) + (bigM - (_Cj - 1)) * (1 - dummy4dwcost[_j]) )
        opt_mdl.addConstr( _Nj >= _Cj * (1 - dummy4dwcost[_j]) )

    rescale_cost = grb.quicksum(dummy4upcost[_j] * res_up[_j] * c_rate[_j] + dummy4dwcost[_j] * res_dw[_j] * c_rate[_j] for _j in J)
    
    # new steady performance 
    steady_perf = grb.quicksum(Tfwd * S_approx[_j] for _j in J)

    # set objective function
    opt_mdl.setObjective(steady_perf - rescale_cost, grb.GRB.MAXIMIZE)

    # solve model
    opt_mdl.optimize()

    
    sol_map = np.zeros((nJ, nN), dtype=np.int8)
    if opt_mdl.Status == grb.GRB.OPTIMAL:
        for _n in N:
            for _j in J:
                sol_map[_j][_n] = 1 if x[_j, _n].x > 0.5 else 0
        rate = [S_approx[_j].getValue() for _j in J]
        cost = [(dummy4upcost[_j] * res_up[_j] * S_approx[_j] + dummy4dwcost[_j] * res_dw[_j] * S_approx[_j]).getValue() for _j in J]
    else:
        rate, cost = [], []

    return opt_mdl.Status, sol_map, np.array(rate), np.array(cost)

def re_allocate_ndf(cmap, jmin, jmax, Ns, Os, res_up, res_dw):
    
    nJ, nN = cmap.shape
    J = range(nJ)
    N = range(nN)
    assert len(jmin) == nJ and len(jmax) == nJ and len(res_up) == nJ and len(res_dw) == nJ
    
    frule = lambda _min, _max, _q: min(_q, _max) * min(1, max(_q-_min, 0))
    q_jnds= np.array([frule(jmin[_j], jmax[_j], nN // nJ) for _j in range(nJ)]).astype(np.int32)

    # FCFS for leftover
    left_over = nN - sum(q_jnds)
    for _j in range(nJ):
        if left_over <= 0: break
        if q_jnds[_j] < jmax[_j]:
            _inc = min(jmax[_j]-q_jnds[_j], left_over)
            if q_jnds[_j] + _inc < jmin[_j]: continue
            q_jnds[_j] += _inc
            left_over -= _inc

    c_map = cmap.copy()
    # the 1st iteration, release overused nodes
    for _j in range(nJ):
        _cN = c_map[_j].sum()
        _q  = q_jnds[_j]
        if _cN <= _q: continue
        _diff = _cN - _q
        _nidx = np.nonzero(c_map[_j])[0][:_diff]
        for _n in _nidx: c_map[_j, _n] = 0
                    
    # the 2nd iteration, re-allocate
    for _j in range(nJ):
        _cN = c_map[_j].sum()
        _q  = q_jnds[_j]

        if jmin[_j] > _q or _cN == _q: continue
        _inc = _q - _cN

        _nidx = np.where(c_map.sum(axis=0)==0)[0][:_inc]
        for _n in _nidx:
            c_map[_j, _n] = 1
    
    _nJ = c_map.sum(axis=1)
    job_rate = np.array([interp1d4s(Ns[_j], Os[_j], _nJ[_j]) if _nJ[_j] >= jmin[_j] else 0 for _j in range(nJ)])
                             
    _cnJ = cmap.sum(axis=1)
    cj_rate = np.array([interp1d4s(Ns[_j], Os[_j], _cnJ[_j]) if _cnJ[_j] >= jmin[_j] else 0 for _j in range(nJ)])
    cost = [res_up[_j]*cj_rate[_j] if _cnJ[_j]<_nJ[_j] else res_dw[_j]*cj_rate[_j] for _j in range(nJ)]
    cost = np.array(cost)
    cost[_nJ == _cnJ] = 0

    return grb.GRB.OPTIMAL, c_map, job_rate, cost