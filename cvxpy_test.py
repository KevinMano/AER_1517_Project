# Importing libraries
import cvxpy as cp
import numpy as np

# Number of states and inputs
n = 1
m = 1

# Dimensions (as given in the paper)
q = n
r = n + m
f = n**2 + n*m

# Defining optimization variables
W = cp.Variable((n+m,n+m), symmetric = True)
Q = cp.Variable((n,n), symmetric = True)
R = cp.Variable((m,n))
lam = cp.Variable((n**2+n*m,n**2+n*m), diag = True)
gam = cp.Variable()

# Defining parameters
bet = cp.Parameter(pos = True, value = 1)

# Generating random values for testing (will change during actual implementation)
np.random.seed(42)
A = np.random.rand(1,1)
B = np.random.rand(1,1)
Bw = np.random.rand(1,1)
Cz = np.array([[np.random.rand(1,1), 0]]).T
Dz = np.array([[0, np.random.rand(1,1)]]).T
Cq = np.array([[np.random.rand(1,1), 0]]).T
Dq = np.array([[0, np.random.rand(1,1)]]).T
Bp = np.array([[1, 1]])

# Constraints
mat_1 = cp.bmat([[W, Cz@Q + Dz@R],
                [cp.transpose(Cz@Q + Dz@R), Q]])

mat_2 = cp.bmat([[Q, np.zeros((n,q)), np.zeros((n,f)), cp.transpose(A@Q + B@R), cp.transpose(Cq@Q + Dq@R)],
                [np.zeros((q,n)), np.eye(q), np.zeros((q,f)), cp.transpose(Bw), np.zeros((q,f))],
                [np.zeros((f,n)), np.zeros((f,q)), lam, cp.transpose(Bp@lam), np.zeros((f,f))],
                [A@Q + B@R, Bw, Bp@lam, Q, np.zeros((n,f))],
                [Cq@Q + Dq@R, np.zeros((f,q)), np.zeros((f,f)), np.zeros((f,n)), bet*lam]])

constraints = [cp.trace(W) <= gam, mat_1 >> 0, mat_2 >> 0]

# Problem
prob = cp.Problem(cp.Minimize(gam), constraints)
prob.solve(solver=cp.MOSEK)
print("status:", prob.status)
print("optimal value", prob.value)
print("R:", R.value)
print("Q:", Q.value)
