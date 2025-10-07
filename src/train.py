from src.models import train_gbr, train_svr, train_knn, train_vr
from src.optimization import run_optimizer
from src.evaluate import save_results
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np

def run_training(config, X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_full, y_train_full, sheet_id):
    models = {"GBR": train_gbr, "SVR": train_svr, "KNN": train_knn, "VotingRegressor": train_vr}
    results, preds = {}, {}

    # --- Train base models ---
    for name, func in models.items():
        metrics, y_pred = func(X_train_full, y_train_full, X_test, y_test)
        results[name] = {"R2": metrics[0], "MSE": metrics[1], "MAE": metrics[2], "RMSE": metrics[3]}
        preds[name] = y_pred
        print(f"{name:<18} R2={metrics[0]:.4f}")

    # --- Optimization phase ---
    best_sol = run_optimizer(config, X_train, y_train, X_valid, y_valid)
    w = np.abs(best_sol[:3]) / np.sum(np.abs(best_sol[:3]))
    n_est, lr, min_leaf, max_depth = int(best_sol[3]), best_sol[4], int(best_sol[5]), int(best_sol[6])
    C, eps, k = best_sol[7], best_sol[8], int(best_sol[9])

    reg1 = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, min_samples_leaf=min_leaf,
                                     max_depth=max_depth, random_state=1)
    reg2 = SVR(C=C, epsilon=eps)
    reg3 = KNeighborsRegressor(n_neighbors=k)
    opt_vr = VotingRegressor([("GB", reg1), ("SVR", reg2), ("KNN", reg3)], weights=w)
    opt_vr.fit(X_train_full, y_train_full)
    y_pred_opt = opt_vr.predict(X_test)

    r2 = round(r2_score(y_test, y_pred_opt), 5)
    mse = round(mean_squared_error(y_test, y_pred_opt), 5)
    mae = round(mean_absolute_error(y_test, y_pred_opt), 5)
    rmse = round(sqrt(mean_squared_error(y_test, y_pred_opt)), 5)

    results["AOA_VR"] = {"R2": r2, "MSE": mse, "MAE": mae, "RMSE": rmse}
    preds["AOA_VR"] = y_pred_opt

    print(f"AOA_VR        R2={r2:.4f}")

    # --- Save all outputs ---
    save_results(config, results, preds, y_test, sheet_id)