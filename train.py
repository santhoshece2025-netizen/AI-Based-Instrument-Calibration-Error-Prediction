"""Train, evaluate, and save a Random Forest movement predictor."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .arm import forward_kinematics

DATA_PATH = "data/robotic_arm_dataset.csv"
MODEL_PATH = "models/arm_movement_model.joblib"
METRICS_PATH = "outputs/metrics.json"
PLOT_PATH = "outputs/actual_vs_predicted.png"

FEATURES = [
    "theta1", "theta2", "target_x", "target_y",
    "current_x", "current_y", "speed", "error_x", "error_y",
    "target_theta1", "target_theta2"
]
TARGETS = ["next_theta1", "next_theta2"]


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python -m src.dataset --samples 6000` first."
        )
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=180, max_depth=18, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    angle_mae = mean_absolute_error(y_test, pred, multioutput="raw_values")
    angle_rmse = np.sqrt(mean_squared_error(y_test, pred, multioutput="raw_values"))

    actual_xy = np.array([forward_kinematics(a, b)
                          for a, b in y_test[TARGETS].to_numpy()])
    pred_xy = np.array([forward_kinematics(a, b) for a, b in pred])
    position_errors = np.linalg.norm(actual_xy - pred_xy, axis=1)
    metrics = {
        "model": "RandomForestRegressor",
        "samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "joint_angle_mae_deg": {
            "theta1": float(angle_mae[0]), "theta2": float(angle_mae[1])
        },
        "joint_angle_rmse_deg": {
            "theta1": float(angle_rmse[0]), "theta2": float(angle_rmse[1])
        },
        "end_effector_position_mae_m": float(position_errors.mean()),
        "end_effector_position_rmse_m": float(np.sqrt(np.mean(position_errors**2))),
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    dump({"model": model, "features": FEATURES, "targets": TARGETS}, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, ax in enumerate(axes):
        actual = y_test[TARGETS[i]].to_numpy()
        predicted = pred[:, i]
        ax.scatter(actual, predicted, alpha=0.45, s=12)
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(f"Actual {TARGETS[i]} (degrees)")
        ax.set_ylabel(f"Predicted {TARGETS[i]} (degrees)")
        ax.set_title(f"{TARGETS[i]}: actual vs predicted")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}\nSaved metrics: {METRICS_PATH}\nSaved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
