import gurobipy as gp
from gurobipy import GRB

model = gp.Model()

x = model.addVars(2, lb=0, vtype=GRB.CONTINUOUS, name="x")

model.setObjective(3*x[0] + 5*x[1], sense=GRB.MAXIMIZE)

model.addConstr(x[0] <= 4, name="c1")

model.addConstr(2*x[1] <= 12, name="c2")

model.addConstr(3*x[0] + 2*x[1] <= 18, name="c3")

model.optimize()

for var in model.getVars():
    if var.X > 0:
        print(f"{var.varName}: {var.X}")

print(f"Objective: {model.objVal}")


print("Changes in RHS of c2")
c2 = model.getConstrByName("c2")
c2.RHS = 24
model.optimize()
for var in model.getVars():
    if var.X > 0:
        print(f"{var.varName}: {var.X}")
print(f"Objective: {model.objVal}")
