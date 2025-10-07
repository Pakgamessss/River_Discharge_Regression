import pandas as pd
from pathlib import Path

def save_results(config, results, preds, y_test, sheet_id):
    out_cfg = config["output"]
    data_name = config["data"]["name"]

    # Prepare directories
    metrics_dir = Path(out_cfg["metrics_dir"])
    preds_dir = Path(out_cfg["predictions_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # File naming pattern: Dataset_Sheet#.ext
    base_name = f"{data_name}_Sheet{sheet_id}"

    # --- Save Metrics ---
    metrics_df = pd.DataFrame(results).T.round(4)
    metrics_df.index.name = "Model"

    if "xlsx" in out_cfg["save_format"]:
        metrics_df.to_excel(metrics_dir / f"{base_name}_metrics.xlsx")
    if "csv" in out_cfg["save_format"]:
        metrics_df.to_csv(metrics_dir / f"{base_name}_metrics.csv")

    # Optional PNG export
    try:
        import dataframe_image as dfi
        if "png" in out_cfg["save_format"]:
            dfi.export(metrics_df, metrics_dir / f"{base_name}_metrics.png")
    except ImportError:
        print("⚠️ dataframe_image not installed — skipping PNG export.")

    # --- Save Predictions ---
    df_pred = pd.DataFrame({"Real": y_test})
    for model_name, preds_arr in preds.items():
        df_pred[model_name] = preds_arr

    df_pred.to_excel(preds_dir / f"{base_name}_predictions.xlsx", index=False)

    if out_cfg["verbose"]:
        print(f"Results saved for {data_name} (Sheet {sheet_id})")
