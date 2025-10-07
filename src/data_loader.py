import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_dataset(config, sheet_id):
    data_cfg = config["data"]
    df = pd.read_excel(data_cfg["file_path"], sheet_name=str(sheet_id)).dropna().iloc[1:, :]

    X_train = df.iloc[:data_cfg["train_end"], :-1]
    y_train = df.iloc[:data_cfg["train_end"], -1]
    X_valid = df.iloc[data_cfg["train_end"]:data_cfg["valid_end"], :-1]
    y_valid = df.iloc[data_cfg["train_end"]:data_cfg["valid_end"], -1]
    X_test = df.iloc[data_cfg["valid_end"]:, :-1]
    y_test = df.iloc[data_cfg["valid_end"]:, -1]
    X_train_full = df.iloc[:data_cfg["valid_end"], :-1]
    y_train_full = df.iloc[:data_cfg["valid_end"], -1]

    return X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_full, y_train_full

def scale_data(config, X_train, X_valid, X_test, X_train_full):
    method = config["scaler"]["method"]
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    X_train_full = scaler.fit_transform(X_train_full)
    return X_train, X_valid, X_test, X_train_full
