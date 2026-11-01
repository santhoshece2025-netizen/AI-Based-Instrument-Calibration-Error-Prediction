"""Generate a reproducible synthetic motion dataset for a 2-link arm."""
import argparse
import os
import numpy as np
import pandas as pd
from .arm import forward_kinematics, inverse_kinematics

RNG_SEED = 42


def generate_dataset(samples: int = 6000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        # Keep joint states within a practical planar workspace.
        theta1 = rng.uniform(-80, 80)
        theta2 = rng.uniform(-115, 115)
        x_now, y_now = forward_kinematics(theta1, theta2)

        # Select a reachable target by generating target joint angles first.
        target_t1 = rng.uniform(-75, 75)
        target_t2 = rng.uniform(-110, 110)
        target_x, target_y = forward_kinematics(target_t1, target_t2)
        speed = rng.uniform(0.08, 1.0)

        # One-step movement fraction is speed-dependent; noisy dynamics make
        # the prediction problem non-trivial but deterministic by seed.
        alpha = 0.08 + 0.52 * speed
        next_t1 = theta1 + alpha * (target_t1 - theta1) + rng.normal(0, 1.0)
        next_t2 = theta2 + alpha * (target_t2 - theta2) + rng.normal(0, 1.0)

        rows.append({
            "theta1": theta1, "theta2": theta2,
            "target_x": target_x, "target_y": target_y,
            "current_x": x_now, "current_y": y_now,
            "speed": speed,
            "error_x": target_x - x_now, "error_y": target_y - y_now,
            "target_theta1": target_t1, "target_theta2": target_t2,
            "next_theta1": next_t1, "next_theta2": next_t2,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--output", default="data/robotic_arm_dataset.csv")
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("--samples must be at least 100")
    df = generate_dataset(args.samples, args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} samples to {args.output}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
