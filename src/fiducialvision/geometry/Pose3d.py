import numpy as np
import numpy.typing as npt

from .Rotation3d import Rotation3d


class Pose3d:
    def __init__(self, x: float, y: float, z: float, rotation: Rotation3d):
        self.x = x
        self.y = y
        self.z = z
        self.rotation = rotation

    def get_translation(self) -> npt.NDArray[np.float64]:
        return np.array([self.x, self.y, self.z])

    def get_rotation(self) -> Rotation3d:
        return self.rotation
