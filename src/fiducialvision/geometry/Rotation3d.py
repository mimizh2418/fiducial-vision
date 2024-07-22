import math

import numpy as np
import numpy.typing as npt


class Rotation3d:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_euler_angles(cls, roll_rads: float, pitch_rads: float, yaw_rads: float):
        cr = math.cos(roll_rads / 2)
        sr = math.sin(roll_rads / 2)

        cp = math.cos(pitch_rads / 2)
        sp = math.sin(pitch_rads / 2)

        cy = math.cos(yaw_rads / 2)
        sy = math.sin(yaw_rads / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(w, x, y, z)

    @classmethod
    def from_rvec(cls, rvec: npt.NDArray[np.float64]):
        angle_rads = np.linalg.norm(rvec)
        if angle_rads == 0:
            return cls(1, 0, 0, 0)

        v = rvec * math.sin(angle_rads / 2) / angle_rads
        return cls(math.cos(angle_rads / 2), v[0], v[1], v[2])

    def get_quaternion(self) -> npt.NDArray[np.float64]:
        return np.array([self.w, self.x, self.y, self.z])

    def get_roll(self) -> float:
        sr_cp = 2 * (self.w * self.x + self.y * self.z)
        cr_cp = 1 - 2 * (self.x * self.x + self.y * self.y)
        return math.atan2(sr_cp, cr_cp)

    def get_pitch(self) -> float:
        sp = math.sqrt(1 + 2 * (self.w * self.y - self.x * self.z))
        cp = math.sqrt(1 - 2 * (self.w * self.y - self.x * self.z))
        return 2 * math.atan2(sp, cp) - math.pi / 2

    def get_yaw(self) -> float:
        sy_cp = 2 * (self.w * self.z + self.x * self.y)
        cy_cp = 1 - 2 * (self.y * self.y + self.z * self.z)
        return math.atan2(sy_cp, cy_cp)
