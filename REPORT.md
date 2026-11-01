# Project Report: Intelligent Robotic Arm Movement Prediction Using Machine Learning

## 1. Abstract
This project explores machine-learning-based prediction of a robotic arm's next joint angles. A synthetic dataset is generated for a two-link planar arm. A Random Forest regression model learns the relationship between current joint state, target coordinates, movement speed, and the desired next joint state. Model performance is evaluated using mean absolute error (MAE), root mean squared error (RMSE), and end-effector position error.

## 2. Problem Statement
Conventional robotic motion controllers typically rely on explicit kinematic equations and control rules. This project investigates whether a regression model can approximate a local next-step movement command from the current state and a target.

## 3. Objectives
- Generate a reproducible joint-motion dataset.
- Engineer useful position and target-error features.
- Train a multi-output regression model.
- Evaluate joint-angle and end-effector prediction error.
- Visualize the predicted path in a software simulation.

## 4. System Overview
Input state and target → feature construction → trained Random Forest → predicted next joint angles → forward kinematics → trajectory and error evaluation.

## 5. Methodology
1. Generate synthetic samples for a planar two-link arm.
2. Compute current and target end-effector coordinates using forward kinematics.
3. Simulate a bounded step toward target joint angles with small noise.
4. Split the data into 80% training and 20% testing subsets.
5. Train `RandomForestRegressor` with 180 trees.
6. Compute angle MAE/RMSE and end-effector position MAE/RMSE.
7. Save plots, metrics, and the fitted model.

## 6. Dataset Description
The generated CSV contains current joint angles, target coordinates, current end-effector coordinates, speed, Cartesian target error, target joint angles, and next joint-angle labels. It is synthetic data generated from a known simulation rule; it is not collected from a physical robot.

## 7. Software and Tools
- Python 3.10+
- NumPy and pandas
- scikit-learn
- Matplotlib
- Joblib

## 8. Results
Run the training script and insert the values produced in `outputs/metrics.json` here. Include `outputs/actual_vs_predicted.png` and `outputs/trajectory_demo.png` in the final presentation. Do not report invented performance figures.

## 9. Conclusion
The project demonstrates a complete workflow for synthetic robotic-motion data generation, supervised regression, numerical evaluation, and trajectory visualization. The model's usefulness depends on data quality and similarity to real robot behaviour.

## 10. Future Enhancements
- Record real joint encoder and end-effector data.
- Compare Random Forest with MLP and gradient-boosting models.
- Add temporal history, acceleration, and actuator limits.
- Integrate with PyBullet or CoppeliaSim.
- Add collision checking and a safety supervisor before any hardware deployment.
