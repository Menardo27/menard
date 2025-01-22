import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Input Data
load_forecast = np.array([
    4, 4, 4, 4, 4, 4, 6, 6, 12, 12, 12, 12, 12, 4, 4, 4, 4, 16, 16, 16, 16, 6.5, 6.5, 6.5
])
solar_forecast = np.array([
    0, 0, 0, 0, 0, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 3.5, 2.5, 2.0, 1.5, 1.0, 0.5, 0, 0, 0, 0, 0, 0
])

thermal_units = ["gen1", "gen2", "gen3"]

# Costs and Parameters
n_units = len(thermal_units)
n_time = len(load_forecast)

thermal_costs = {
    "a": np.array([5.0, 5.0, 5.0]),
    "b": np.array([0.5, 0.5, 3.0]),
    "c": np.array([1.0, 0.5, 2.0]),
    "startup": np.array([2.0, 2.0, 2.0]),
    "shutdown": np.array([1.0, 1.0, 1.0]),
}

pmin = np.array([1.5, 2.5, 1.0])
pmax = np.array([5.0, 10.0, 3.0])
init_status = np.array([0, 0, 0])

# Create the model
with gp.Env() as env, gp.Model(env=env) as model:
    # Create variables as matrices
    power = model.addMVar((n_units, n_time), lb=0, name="power")
    commitment = model.addMVar((n_units, n_time), vtype=GRB.BINARY, name="commitment")
    startup = model.addMVar((n_units, n_time), vtype=GRB.BINARY, name="startup")
    shutdown = model.addMVar((n_units, n_time), vtype=GRB.BINARY, name="shutdown")

    # Objective function
    fixed_cost = commitment @ thermal_costs["a"]
    linear_cost = power @ thermal_costs["b"]
    quadratic_cost = (power ** 2) @ thermal_costs["c"]
    startup_cost = startup @ thermal_costs["startup"]
    shutdown_cost = shutdown @ thermal_costs["shutdown"]

    total_cost = (
        gp.quicksum(fixed_cost)
        + gp.quicksum(linear_cost)
        + gp.quicksum(quadratic_cost)
        + gp.quicksum(startup_cost)
        + gp.quicksum(shutdown_cost)
    )
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints
    # Power balance (matrix operation)
    model.addConstr(
        power.sum(axis=0) + solar_forecast == load_forecast, name="power_balance"
    )

    # Logical constraints for commitment
    # First time step
    model.addConstr(
        commitment[:, 0] - init_status == startup[:, 0] - shutdown[:, 0],
        name="logical_first_step",
    )

    # Subsequent time steps
    model.addConstr(
        commitment[:, 1:] - commitment[:, :-1]
        == startup[:, 1:] - shutdown[:, 1:],
        name="logical_steps",
    )

    # Startup and shutdown cannot happen simultaneously
    model.addConstr(
        startup + shutdown <= 1, name="startup_shutdown_conflict"
    )

    # Physical constraints with indicator constraints
    for g in range(n_units):
        for t in range(n_time):
            model.addGenConstrIndicator(
                commitment[g, t], True, power[g, t] >= pmin[g]
            )
            model.addGenConstrIndicator(
                commitment[g, t], True, power[g, t] <= pmax[g]
            )
            model.addGenConstrIndicator(
                commitment[g, t], False, power[g, t] == 0
            )

    # Optimize
    model.optimize()

    # Display results
    if model.status == GRB.OPTIMAL:
        print(f"Total Cost: {model.ObjVal}")
        print("\nPower Output:")
        print(power.X)
        print("\nCommitment Status:")
        print(commitment.X)
