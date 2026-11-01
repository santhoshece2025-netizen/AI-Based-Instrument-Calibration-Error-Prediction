"""Show a short target-seeking trajectory using the trained predictor."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import load
from .arm import forward_kinematics


def main():
    path = "models/arm_movement_model.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Train first: python -m src.train")
    bundle = load(path)
    model, features = bundle["model"], bundle["features"]

    theta1, theta2 = -35.0, 55.0
    target_theta1, target_theta2 = 45.0, -50.0
    tx, ty = forward_kinematics(target_theta1, target_theta2)
    speed = 0.65
    history = [(theta1, theta2)]
    for _ in range(25):
        x, y = forward_kinematics(theta1, theta2)
        row = {
            "theta1": theta1, "theta2": theta2,
            "target_x": tx, "target_y": ty,
            "current_x": x, "current_y": y,
            "speed": speed, "error_x": tx-x, "error_y": ty-y,
            "target_theta1": target_theta1, "target_theta2": target_theta2
        }
        prediction = model.predict([[row[name] for name in features]])[0]
        theta1, theta2 = float(prediction[0]), float(prediction[1])
        history.append((theta1, theta2))

    points = np.array([forward_kinematics(a, b) for a, b in history])
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(points[:, 0], points[:, 1], marker="o", markersize=3, label="Predicted path")
    ax.scatter([tx], [ty], marker="*", s=180, label="Target")
    ax.scatter([points[0, 0]], [points[0, 1]], marker="s", s=60, label="Start")
    ax.set_title("2-Link Robotic Arm: Predicted End-Effector Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/trajectory_demo.png", dpi=160)
    plt.close(fig)
    print("Saved outputs/trajectory_demo.png")
    print(f"Final predicted end-effector position: ({points[-1,0]:.3f}, {points[-1,1]:.3f}) m")
    print(f"Target position: ({tx:.3f}, {ty:.3f}) m")


if __name__ == "__main__":
    main()
