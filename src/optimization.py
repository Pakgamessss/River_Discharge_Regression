import numpy as np
from mealpy import FloatVar, AOA, HHO, HGS
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# --- Objective Function ---
def objective_function(solution, X_train, y_train, X_valid, y_valid):
    w = np.abs(solution[:3]) / np.sum(np.abs(solution[:3]))
    n_est, lr, min_leaf, max_depth = int(solution[3]), solution[4], int(solution[5]), int(solution[6])
    C, eps, k = solution[7], solution[8], int(solution[9])

    reg1 = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, min_samples_leaf=min_leaf,
                                     max_depth=max_depth, random_state=1)
    reg2 = SVR(C=C, epsilon=eps)
    reg3 = KNeighborsRegressor(n_neighbors=k)

    model = VotingRegressor([("GB", reg1), ("SVR", reg2), ("KNN", reg3)], weights=w)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return -r2_score(y_valid, preds)  # negative for minimization

# --- Run Optimizer ---
def run_optimizer(config, X_train, y_train, X_valid, y_valid):
    opt_cfg = config["optimizer"]
    lb, ub = np.array(opt_cfg["bounds"]["lower"]), np.array(opt_cfg["bounds"]["upper"])
    n_vars = len(lb)

    bounds = FloatVar(lb=tuple(lb), ub=tuple(ub), name="params")

    problem = {
        "bounds": bounds,
        "minmax": "min",
        "obj_func": lambda s: objective_function(s, X_train, y_train, X_valid, y_valid),
    }

    algo = opt_cfg["algorithm"].upper()
    epochs = opt_cfg["epochs"]
    pop = opt_cfg["population_size"]

    if algo == "AO":
        model = AOA.OriginalAOA(epoch=epochs, pop_size=pop, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9)
    elif algo == "HHO":
        model = HHO.BaseHHO(epoch=epochs, pop_size=pop)
    elif algo == "HGS":
        model = HGS.OriginalHGS(epoch=epochs, pop_size=pop)
    else:
        raise ValueError(f"Unsupported optimizer: {algo}")

    g_best = model.solve(problem)
    print(f"Best Fitness (R²): {-g_best.target.fitness:.4f}")
    print(f"Best Solution: {g_best.solution}")

    return g_best.solution
