# Intelligent Robotic Arm Movement Prediction Using Machine Learning

A complete, reproducible capstone project that predicts a robotic arm's next joint angles and end-effector position from recent motion state and target coordinates. It generates a synthetic dataset, trains a Random Forest regression model, evaluates it with MAE/RMSE, plots actual-vs-predicted results, and optionally demonstrates the predicted trajectory in a simple 2D simulation.

> **Safety / scope:** This project uses a synthetic 2-link planar arm and a software simulation. It does not control physical hardware. If adapting it to a real arm, add hard joint limits, emergency stop, low-speed testing, collision checks, and independent safety interlocks.

## Features

- Synthetic joint-motion dataset with target coordinates, joint angles, movement speed, and end-effector coordinates.
- Feature engineering from current joint state, target, target error, and speed.
- `RandomForestRegressor` multi-output prediction of next joint angles.
- Forward kinematics converts predicted joint angles into end-effector coordinates.
- MAE/RMSE evaluation, joint-angle errors, position error, and plots.
- Saved trained model and metrics.
- Small 2D trajectory demo.

## Project structure

```text
intelligent_robotic_arm_ml/
├── data/                  # generated CSV dataset
├── models/                # trained model (created by training)
├── outputs/               # metrics and plots
├── src/
│   ├── arm.py              # planar arm kinematics
│   ├── dataset.py          # synthetic dataset generator
│   ├── train.py            # training and evaluation
│   └── simulate.py         # predicted trajectory demonstration
├── tests/
│   └── test_arm.py
├── requirements.txt
└── .gitignore
```

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python -m src.dataset --samples 6000
python -m src.train
python -m src.simulate
```

The scripts create:
- `data/robotic_arm_dataset.csv`
- `models/arm_movement_model.joblib`
- `outputs/metrics.json`
- `outputs/actual_vs_predicted.png`
- `outputs/trajectory_demo.png`

## Dataset

The synthetic dataset is generated from a two-link planar arm. Each sample represents a current state and a nearby desired target. The next joint angles are calculated with a bounded, noisy proportional step toward inverse-kinematic target angles. Features include:

- current joint angles (degrees)
- target end-effector x/y (m)
- current end-effector x/y (m)
- movement speed (normalized)
- target error x/y (m)
- joint-angle difference to the target (degrees)

Targets are the next joint angles. This is a simulation dataset, not measured hardware data; clearly state this in the report.

## Model and evaluation

A `RandomForestRegressor` predicts both next joint angles. Evaluation reports:
- joint angle MAE and RMSE (degrees)
- end-effector position MAE and RMSE (metres), after forward kinematics
- a test-set scatter plot comparing actual and predicted joint angles

The dataset is split into train/test sets with a fixed seed for reproducibility.

## GitHub upload

1. Unzip this archive.
2. Create a new GitHub repository, e.g. `intelligent-robotic-arm-ml`.
3. Upload the extracted files, or run:

```bash
git init
git add .
git commit -m "Initial robotic arm movement prediction project"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/intelligent-robotic-arm-ml.git
git push -u origin main
```

Replace `<YOUR-USERNAME>` with your GitHub username. Do not upload `.venv/` or other local environment folders.

## Limitations and future work

- Synthetic data may not represent a real robot's dynamics, backlash, friction, or sensor noise.
- The current model predicts a local next-step joint command, not a guaranteed collision-free path.
- Improve by collecting real joint encoder data, adding temporal sequence features, comparing MLP/gradient boosting models, and integrating with PyBullet or CoppeliaSim.
