import numpy as np
import time

def frag_sampling(n):
    ref_segs = np.load('fragments.npy')
    bins = (30, 62, 120, 180, 240, 300, 360, 420, 480, 540, \
            600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300,\
            3600, 4200, 5400, 6000, 7200, 8400, 9600)[2:-1]

    hist, bin_edge = np.histogram(ref_segs, bins=bins)
    bins_prob = hist / hist.sum()
    
    ret = np.zeros(n)
    for i in range(n):
        bi = np.random.choice(range(len(bins_prob)), 1, p=bins_prob)
        ret[i] = np.random.rand() * (bin_edge[bi+1] - bin_edge[bi]) + bin_edge[bi]
    return ret

def synthetic_trace(nodes, nf):
    np.random.seed(2021)
    fragments = frag_sampling(nf)
    avg_frag = np.mean(fragments)
    
    trace_cont = {_n:[] for _n in nodes}
    for i in range(nf):
        _nd = np.random.randint(0, len(nodes)) # node id
        _oc = np.random.poisson(10*avg_frag)   # occupied time
        _oc = 120 # TODO: change back to 120 after debugging
        if len(trace_cont[nodes[_nd]]) == 0:
            pfe = 0
        else:
            pfe = trace_cont[nodes[_nd]][-1][-1] # end of last fragment
        trace_cont[nodes[_nd]].append((pfe + _oc, pfe + _oc + fragments[i])) # frag start and end
    return trace_cont

def create_events_base_on_trace(nodes, nf):
    trace_cont = synthetic_trace(nodes, nf=20000)

    # for record the nodes status
    counters = {}
    flags = {}
    for node in nodes:
        counters[node] = 0
        flags[node] = False

    start_time = time.time()

    while(True):
        time.sleep(0.1)
        relate_time = time.time() - start_time

        for node in nodes:
            timestamps_tuple = trace_cont[node] # list of tuple for node
            current_time_tuple = timestamps_tuple[counters[node]]

            if relate_time > current_time_tuple[0] and flags[node] == False:
                print("node " + node + " in") # trigger node in event
                flags[node] = True

            if relate_time > current_time_tuple[1] and flags[node] == True:
                print("node " + node + " leave") # trigger node leave event
                flags[node] = False
                counters[node] = counters[node] + 1
