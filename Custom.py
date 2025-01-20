from functools import partial

import gurobipy as gp
from gurobipy import GRB


class CallbackData:
    def __init__(self):
        self.last_gap_change_time = -GRB.INFINITY
        self.last_gap = GRB.INFINITY

def callback(model, where, *, cbdata):
    if where != GRB.Callback.MIP:
        return
    
    # Check if at least one incumbent solution has been found
    if model.cbGet(GRB.Callback.MIP_SOLCNT) == 0:
        return

    # Current runtime and MIPGap
    runtime = model.cbGet(GRB.Callback.RUNTIME)
    mip_gap = model.cbGet(GRB.Callback.MIP_OBJBST) - model.cbGet(GRB.Callback.MIP_OBJBND)

    # If this is the first incumbent solution, initialize the timer and gap
    if cbdata.last_gap_change_time == -GRB.INFINITY:
        cbdata.last_gap_change_time = runtime
        cbdata.last_gap = mip_gap
        return

    # Check if the MIPGap has decreased sufficiently
    if cbdata.last_gap - mip_gap > epsilon_to_compare_gap:
        cbdata.last_gap_change_time = runtime
        cbdata.last_gap = mip_gap
    
    # Terminate if no significant improvement in the gap within the time limit
    if runtime - cbdata.last_gap_change_time > time_from_best:
        print("Terminating optimization due to lack of sufficient gap improvement.")
        model.terminate()

with gp.read("data/mkp.mps.bz2") as model:
    # Global variables used in the callback function
    time_from_best = 50  # Time in seconds
    epsilon_to_compare_gap = 1e-4

    # Initialize data passed to the callback function
    callback_data = CallbackData()
    callback_func = partial(callback, cbdata=callback_data)

    # Optimize the model with the custom callback
    model.optimize(callback_func)
