# omega_optimization.py
import cvxpy as cp
import numpy as np
from config import *

def worst_case_omega_optimization(predicted_returns, historical_returns, predictive_errors, n_stocks):
    x = cp.Variable(n_stocks)
    psi = cp.Variable()
    u = cp.Variable((SAMPLE_PERIOD, NUM_DISTRIBUTIONS))
    
    # Objective function
    obj = WEIGHTS[0] * psi
    obj += WEIGHTS[1] * cp.sum(x @ predicted_returns)
    
    # Corrected predictive errors term
    mean_portfolio_errors = [cp.mean(x @ predictive_errors[:, :, j].T) for j in range(NUM_DISTRIBUTIONS)]
    obj += WEIGHTS[2] * cp.sum(mean_portfolio_errors)
    
    obj += WEIGHTS[3] * cp.sum(x @ historical_returns[:, 0])
    obj += WEIGHTS[4] * cp.sum(x @ historical_returns[:, 1])
    
    # Constraints
    constraints = []
    for j in range(NUM_DISTRIBUTIONS):
        portfolio_return = x @ predicted_returns
        portfolio_loss = (1 - DELTA) / SAMPLE_PERIOD * cp.sum(u[:, j])
        constraints.append(DELTA * portfolio_return - portfolio_loss >= psi)
        for t in range(SAMPLE_PERIOD):
            constraints.append(u[t, j] >= -x @ predictive_errors[t, :, j])
            constraints.append(u[t, j] >= 0)
    constraints.append(cp.sum(x) == 1)
    constraints.append(x >= 0)
    constraints.append(x <= 1)
    
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve()
    
    return x.value if prob.status == 'optimal' else np.ones(n_stocks) / n_stocks