import json
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Charger les données
with open("data/portfolio-example.json", "r") as f:
    data = json.load(f)

n = data["num_assets"]
sigma = np.array(data["covariance"])
mu = np.array(data["expected_return"])
mu_0 = data["target_return"]
k = data["portfolio_max_size"]

# Créer le modèle
model = gp.Model("portfolio")

# Variables
w = model.addVars(n, lb=0, ub=1, name="w")  # Poids des actifs
x = model.addVars(n, vtype=GRB.BINARY, name="x")  # Inclusion des actifs

# Fonction objectif : minimiser le risque (variance)
model.setObjective(gp.quicksum(w[i] * sigma[i, j] * w[j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

# Contraintes
model.addConstr(gp.quicksum(w[i] * mu[i] for i in range(n)) >= mu_0, name="return")  # Retour minimum
model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1, name="budget")  # Somme des poids = 1
model.addConstr(gp.quicksum(x[i] for i in range(n)) <= k, name="max_size")  # Taille max du portefeuille

# Assurer que w[i] <= x[i]
for i in range(n):
    model.addConstr(w[i] <= x[i], name=f"link_{i}")

# Optimiser
model.optimize()

# Extraire la solution
if model.Status == GRB.OPTIMAL:
    portfolio = [w[i].X for i in range(n)]
    selected_assets = [i for i in range(n) if x[i].X > 0.5]
    risk = model.ObjVal
    expected_return = sum(w[i].X * mu[i] for i in range(n))

    # Créer un DataFrame pour la solution
    df = pd.DataFrame(
        data=portfolio + [risk, expected_return],
        index=[f"asset_{i}" for i in range(n)] + ["risk", "return"],
        columns=["Portfolio"],
    )

    print("Optimal Portfolio:")
    print(df)
    print("Selected assets:", selected_assets)
else:
    print("No optimal solution found.")
