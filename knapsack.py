import numpy as np
import gurobipy as gp
from gurobipy import GRB

def generate_knapsack(num_items):
    # Fix seed value
    rng = np.random.default_rng(seed=0)
    # Item values, weights
    values = rng.uniform(low=1, high=25, size=num_items)
    weights = rng.uniform(low=5, high=100, size=num_items)
    # Knapsack capacity
    capacity = 0.7 * weights.sum()

    return values, weights, capacity


def solve_knapsack_model(values, weights, capacity):
    num_items = len(values)
    # Turn values and weights numpy arrays to dict
    values_dict = {i: values[i] for i in range(num_items)}
    weights_dict = {i: weights[i] for i in range(num_items)}

    with gp.Env() as env:
        with gp.Model(name="knapsack", env=env) as model:
            # Define decision variables using the Model.addVars() method
            x = model.addVars(num_items, vtype=GRB.BINARY, name="x")

            # Define objective function using the Model.setObjective() method
            # Build the LinExpr using the tupledict.prod() method
            model.setObjective(x.prod(values_dict), GRB.MAXIMIZE)

            # Define capacity constraint using the Model.addConstr() method
            model.addConstr(x.prod(weights_dict) <= capacity, name="capacity")

            model.optimize()

            # Check if an optimal solution was found
            if model.status == GRB.OPTIMAL:
                # Extract the selected items
                selected_items = [i for i in range(num_items) if x[i].x > 0.5]
                total_value = model.objVal
                total_weight = sum(weights[i] for i in selected_items)
                print(f"Optimal solution found!")
                print(f"Total value: {total_value}")
                print(f"Total weight: {total_weight}")
                print(f"Selected items: {selected_items}")
            else:
                print("No optimal solution found.")

data = generate_knapsack(10000)
solve_knapsack_model(*data)