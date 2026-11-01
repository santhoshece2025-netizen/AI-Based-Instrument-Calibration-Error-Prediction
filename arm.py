"""Kinematics helpers for a two-link planar robotic arm."""
from math import cos, sin, radians, degrees, atan2, acos, sqrt

LINK_1 = 0.50  # metres
LINK_2 = 0.35  # metres


def forward_kinematics(theta1_deg: float, theta2_deg: float,
                       link1: float = LINK_1, link2: float = LINK_2):
    """Return end-effector (x, y) in metres for joint angles in degrees."""
    t1, t2 = radians(theta1_deg), radians(theta2_deg)
    x = link1 * cos(t1) + link2 * cos(t1 + t2)
    y = link1 * sin(t1) + link2 * sin(t1 + t2)
    return float(x), float(y)


def inverse_kinematics(x: float, y: float, elbow: int = 1,
                       link1: float = LINK_1, link2: float = LINK_2):
    """Return one reachable 2-link IK solution in degrees, or None."""
    r2 = x*x + y*y
    c2 = (r2 - link1*link1 - link2*link2) / (2*link1*link2)
    if c2 < -1.0 or c2 > 1.0:
        return None
    c2 = max(-1.0, min(1.0, c2))
    t2 = (1 if elbow >= 0 else -1) * acos(c2)
    t1 = atan2(y, x) - atan2(link2*sin(t2), link1 + link2*cos(t2))
    return degrees(t1), degrees(t2)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
