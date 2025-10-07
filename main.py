import yaml
from src.data_loader import load_dataset, scale_data
from src.train import run_training

def main():
    with open("./config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    for sheet_id in range(config["data"]["sheet_range"][0], config["data"]["sheet_range"][1]):
        print(f"\n=== Scenario {sheet_id} ===")
        X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_full, y_train_full = load_dataset(config, sheet_id)
        X_train, X_valid, X_test, X_train_full = scale_data(config, X_train, X_valid, X_test, X_train_full)
        run_training(config, X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_full, y_train_full, sheet_id)

if __name__ == "__main__":
    main()
