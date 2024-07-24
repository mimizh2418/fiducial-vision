import numpy as np
import numpy.typing as npt
from wpimath.geometry import Pose3d, Rotation3d, Translation3d


def to_opencv_translation(translation: Translation3d) -> npt.NDArray[np.float64]:
    return np.array([-translation.y, -translation.z, translation.x])


def from_opencv_translation(tvec: npt.NDArray[np.float64]) -> Translation3d:
    return Translation3d(tvec[2], -tvec[0], -tvec[1])


def from_opencv_rotation(rvec: npt.NDArray[np.float64]) -> Rotation3d:
    return Rotation3d(np.array([rvec[2], -rvec[0], -rvec[1]]), np.linalg.norm(rvec))
