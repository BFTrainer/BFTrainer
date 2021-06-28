# BFTrainer
This repository officially hosts the implementation of `BFTrainer: Low-Cost Training of Neural Networks on UnfillableSupercomputer Nodes` [https://arxiv.org/abs/2106.12091](https://arxiv.org/abs/2106.12091).

The BFTrainer's philosophy of design is **zero-cost** to jobs in the main scheduler of the supercomputer. It does mean that BFTrainer is the lowest priority to use resource but it does NOT mean that jobs run by BFTrainer will be slow because BFTrainer **relaxes** resource requirement and optimizes the way to utilize any fragmented resource. Depending on relative demand, BFTrainer may run slower or faster than main queue.

## Main Scheduler 
In theory BFTrainer works with any batch scheduler that has either one of the follow features: 
- Grand SSH access to idle nodes for particular users, e.g., BFTrainer admin/user, the scheduler can revoke (i.e., pre-empt) at any time.
- shrink-to-fit, i.e., the scheduler allows submitting a request with a range of nodes and later stretch the allocation when needed (i.e., pre-empt) until all nodes are preempted or request timeout.

We tested our implementation with OpenPBS.

## User's Code decoration 
We try to minimize code changes, user need to:
- Implement their DNN training using Elastic Horovod, it's oretty straightforward, instructions available [here](https://horovod.readthedocs.io/en/stable/elastic_include.html).
- User only need to add a line of code in ther iterative training loop to report the progress (e.g., samples processed) that they have made from the iteration.

# MILP solver for optimal resource allocation
We provided two implementations of the MILP model:
- [PYTHON-MIP](https://www.python-mip.com) with open source (free) [CBC](https://github.com/coin-or/Cbc) solver.
- [gurobipy](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_grbpy_the_gurobi_python.html), the Gurobi Python Interface, with [Gurobi](https://www.gurobi.com) solver. Licensed required, one can use free trial or academia license for evaluation purpose.

Based on our preliminary benchmark, Gurobi is much faster than CBC when problem size is big (e.g., dozens of jobs on hundreds of nodes).
Otherwise, the time to solve is very similar between Gurobi and CBC when problem size is small.<br>

The PYTHON-MIP also supports using Gurobi as solver (license required as well) but slower than gurobipy in most cases especially when problem size is large.
Thus, one can use the free CBC when problem size is small, especailly for relatively small supercomputers. 
Otherwise, Gurobi with the gurobipy based implementation is recommended.
