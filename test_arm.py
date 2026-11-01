import unittest
from src.arm import forward_kinematics, inverse_kinematics


class TestArmKinematics(unittest.TestCase):
    def test_zero_angles(self):
        x, y = forward_kinematics(0, 0)
        self.assertAlmostEqual(x, 0.85, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_inverse_forward_round_trip(self):
        target = forward_kinematics(25, 40)
        solution = inverse_kinematics(*target, elbow=1)
        self.assertIsNotNone(solution)
        x2, y2 = forward_kinematics(*solution)
        self.assertAlmostEqual(target[0], x2, places=6)
        self.assertAlmostEqual(target[1], y2, places=6)


if __name__ == "__main__":
    unittest.main()
