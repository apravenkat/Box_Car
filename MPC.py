import cvxpy as cp
import numpy as np
class LongitudinalMPC:
    def __init__(self, dt = 1/50, N=15, a_max = 6, a_min = -6):
        self.dt = dt
        self.N = N
        self.a_max = a_max
        self.a_min = a_min
        self.F_max = 400
        self.mass = 400
        self.prev_error = 0
        self.errors = 0
    
    def solve(self, v0, v_ref_seq = None, v_min_seq = None, v_max_seq = None):
        N = self.N
        dt = self.dt

        a = cp.Variable(N)
        v = cp.Variable(N+1)
        cost = 0
        constraints = []

        constraints += [v[0] == v0]

        Qv = 10
        Ru = 0.1
        Rdu = 1

        for k in range(N):
            constraints += [v[k+1] == v[k] + a[k]*dt]
            constraints += [a[k] <= self.a_max, a[k] >= self.a_min]

            if v_min_seq is not None:
                constraints += [v[k+1] >= v_min_seq[k] >= v_min_seq[k]]
            if v_max_seq is not None:
                constraints += [v[k+1] <= v_max_seq[k] <= v_max_seq[k]]

            cost += Qv * cp.square(v[k+1] - v_ref_seq[k]) + Ru * cp.square(a[k])

            if k>0:
                cost += Rdu * cp.square(a[k] - a[k-1])
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP,warm_start=True,verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            a_cmd = np.clip((v_ref_seq[0]-v0)/dt, self.a_min, self.a_max)
            throttle
