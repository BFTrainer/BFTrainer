import time
import numpy as np
from scipy import interpolate
from mip import Model, xsum, maximize, BINARY, CONTINUOUS, INTEGER, OptimizationStatus, CBC
import pandas as pd

def interp1d4s(Ns, Os, Nx):
    f = interpolate.interp1d(Ns, Os)
    return float(f(Nx))

def re_allocate(cmap, jmin, jmax, Ns, Os, Tfwd, res_up, res_dw, time_limit, note, solver=CBC):
    start_time = str(time.time())
    joblist = cmap.index
    nodelist= cmap.columns
    nJ, nN = cmap.shape
    J = range(nJ)
    N = range(nN)
    assert len(jmin) == nJ and len(jmax) == nJ and len(res_up) == nJ and len(res_dw) == nJ
    # compute throughput of current map, cost will depend on it
    _cNj = cmap.sum(axis=1)
    c_rate = [interp1d4s(Ns[_j], Os[_j], _cNj[_j]) \
              if _cNj[_j] >= jmin[_j] else 0 for _j in range(cmap.shape[0])]

    with open("%s-b4.log" % start_time, 'w') as fp:
        fp.write(cmap.to_string() + '\n')
        fp.write(f"jmin={jmin}, jmax={jmax}, Ns={Ns}, Os={Os}, Tfwd={Tfwd}, res_up={res_up}, res_dw={res_dw}, time_limit={time_limit}\n")
        fp.write(note)

    c_map = cmap.values
        
    m = Model(solver_name=solver)
    
    # create decision variable, J x N
    x = [[m.add_var('x({},{})'.format(j, n), var_type=BINARY) for n in N] for j in J]
        
    # job scalability constraint, should be either within the internal or zero
    bigM = nN + 1

    dummy4Jmin = [m.add_var(var_type=BINARY) for _j in J]
    dummy4Jmax = [m.add_var(var_type=BINARY) for _j in J]

    for _j in J:
        _nJ = xsum(x[_j][_n] for _n in N)

        m += _nJ >= jmin[_j] - bigM * dummy4Jmin[_j]
        m += _nJ <= 0 + bigM * (1 - dummy4Jmin[_j])

        m += jmax[_j] >= _nJ - bigM * dummy4Jmax[_j]
        m += _nJ <= 0 + bigM * (1 - dummy4Jmax[_j])

    # node allocations constraint
    for _n in N:
        m += xsum(x[_j][_n] for _j in J) <= 1 # at most one application

    # constraint to disallow job migration 
    dummy4xxorc = [[m.add_var('xxorc({},{})'.format(j, n), var_type=BINARY) for n in N] for j in J] # x XOR c
    dummy4migr  = [m.add_var(var_type=BINARY) for _j in J]
    
    c = c_map # a shorter name for easy ref
    for _j in J:
        for _n in N:
            m += dummy4xxorc[_j][_n] <= x[_j][_n] + c[_j][_n]
            m += dummy4xxorc[_j][_n] >= x[_j][_n] - c[_j][_n]
            m += dummy4xxorc[_j][_n] >= c[_j][_n] - x[_j][_n]
            m += dummy4xxorc[_j][_n] <= 2 - x[_j][_n] - c[_j][_n]

    for _j in J:
        _xxorc_sumJ = xsum(dummy4xxorc[_j][_n] for _n in N)
        _nj_new = xsum(x[_j][_n] for _n in N)
        _nj_old = sum(c[_j])
        m += _nj_new - _nj_old >= _xxorc_sumJ  - bigM * dummy4migr[_j]
        m += _nj_new - _nj_old <= -_xxorc_sumJ + bigM * (1 - dummy4migr[_j])

    # approximate the scalability
    w4sappro = [[m.add_var('x({},{})'.format(j, i), var_type=CONTINUOUS, lb=0, ub=1) for i in range(len(Ns[j]))] for j in J]

    S_approx = {}
    for _j in J:
        I = range(len(Ns[_j]))
        m += xsum(w4sappro[_j]) == 1 # convexification

        _Nj = xsum(x[_j][_n] for _n in N)
        m +=_Nj == xsum(Ns[_j][_i] * w4sappro[_j][_i] for _i in I)

        m.add_sos([(w4sappro[_j][_i], Ns[_j][_i]) for _i in I], 2)  

        S_approx[_j] = xsum(Os[_j][_i] * w4sappro[_j][_i] for _i in I)

    # scale up/down cost
    dummy4upcost = [m.add_var(var_type=BINARY) for _j in J]
    dummy4dwcost = [m.add_var(var_type=BINARY) for _j in J]

    for _j in J:
        _Nj = xsum(x[_j][_n] for _n in N)
        _Cj = sum(c_map[_j])
        m += _Nj <= _Cj + (bigM - _Cj) * dummy4upcost[_j]
        m += _Nj >= (_Cj + 1) * dummy4upcost[_j]

        m += _Nj <= (_Cj - 1) + (bigM - (_Cj - 1)) * (1 - dummy4dwcost[_j])
        m += _Nj >= _Cj * (1 - dummy4dwcost[_j])
    
    rescale_cost = xsum(dummy4upcost[_j] * res_up[_j] * c_rate[_j] + dummy4dwcost[_j] * res_dw[_j] * c_rate[_j] for _j in J)
    
    # new steady performance 
    steady_perf = xsum(Tfwd * S_approx[_j] for _j in J)

    # set objective function
    m.objective = maximize(steady_perf - rescale_cost)

    # solve model
    status = m.optimize(max_seconds=time_limit)

    if status == OptimizationStatus.OPTIMAL:
        print('optimal solution cost {} found'.format(m.objective_value)) 
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound)) 
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(m.objective_bound)) 
    else:
        print('something wrong', status)

    sol_map = np.zeros((nJ, nN), dtype=np.int8)
    if status == OptimizationStatus.OPTIMAL:
        for _n in N:
            for _j in J:
                sol_map[_j][_n] = 1 if x[_j][_n].x > 0.5 else 0
        rate = [S_approx[_j].x for _j in J]
        cost = [(dummy4upcost[_j] * res_up[_j] * c_rate[_j] + dummy4dwcost[_j] * res_dw[_j] * c_rate[_j]) for _j in J]
    else:
        rate, cost = [], []
    
    sol_map_pd = pd.DataFrame(sol_map, index=joblist, columns=nodelist)
    with open("%s-after.log" % start_time, 'w') as fp:
        fp.write(sol_map_pd.to_string() + '\n')
        fp.write(f"opt_mdl.Status={status}, rate={rate}, cost={cost}\n")
        fp.write(note)
        
    return status, sol_map, np.array(rate), np.array(cost)