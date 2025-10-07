from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

def evaluate(y_true, y_pred):
    r2 = round(r2_score(y_true, y_pred), 5)
    mse = round(mean_squared_error(y_true, y_pred), 5)
    mae = round(mean_absolute_error(y_true, y_pred), 5)
    rmse = round(sqrt(mean_squared_error(y_true, y_pred)), 5)
    return r2, mse, mae, rmse

def train_gbr(x_train, y_train, x_test, y_test):
    model = GradientBoostingRegressor(random_state=1)
    model.fit(x_train, y_train)
    return evaluate(y_test, model.predict(x_test)), model.predict(x_test)

def train_svr(x_train, y_train, x_test, y_test):
    model = SVR()
    model.fit(x_train, y_train)
    return evaluate(y_test, model.predict(x_test)), model.predict(x_test)

def train_knn(x_train, y_train, x_test, y_test):
    model = KNeighborsRegressor()
    model.fit(x_train, y_train)
    return evaluate(y_test, model.predict(x_test)), model.predict(x_test)

def train_vr(x_train, y_train, x_test, y_test):
    models = [
        ("GBR", GradientBoostingRegressor(random_state=1)),
        ("SVR", SVR()),
        ("KNN", KNeighborsRegressor())
    ]
    vr = VotingRegressor(estimators=models)
    vr.fit(x_train, y_train)
    return evaluate(y_test, vr.predict(x_test)), vr.predict(x_test)
