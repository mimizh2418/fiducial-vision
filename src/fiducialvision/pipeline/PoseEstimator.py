import sys
from typing import Sequence, List, Union

import cv2
import numpy as np
import numpy.typing as npt
from wpimath.geometry import Pose3d, Rotation3d, Transform3d

from ..config import Config
from ..coordinate_util import to_opencv_translation, from_opencv_translation, from_opencv_rotation
from .pipeline_types import FiducialTagDetection, PoseEstimatorResult


class PoseEstimator:
    config: Config

    def __init__(self, config: Config):
        self.config = config

    def solve_single_target(self, observed_tag: FiducialTagDetection) -> Union[PoseEstimatorResult, None]:
        if (not self.config.has_tag_layout()
                or not self.config.has_calibration()
                or len(self.config.fiducial.tag_layout) == 0
                or observed_tag.id not in self.config.fiducial.tag_layout
                or len(observed_tag.corners) != 4):
            return None

        object_points = np.array([[-self.config.fiducial.tag_size_m / 2.0, self.config.fiducial.tag_size_m / 2.0, 0],
                                  [self.config.fiducial.tag_size_m / 2.0, self.config.fiducial.tag_size_m / 2.0, 0],
                                  [self.config.fiducial.tag_size_m / 2.0, -self.config.fiducial.tag_size_m / 2.0, 0],
                                  [-self.config.fiducial.tag_size_m / 2.0, -self.config.fiducial.tag_size_m / 2.0, 0]])
        image_points = observed_tag.corners

        try:
            retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(object_points,
                                                                      image_points,
                                                                      self.config.calibration.intrinsics_matrix,
                                                                      self.config.calibration.distortion_coefficients,
                                                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error as e:
            print(f"Error in SOLVEPNP_IPPE_SQUARE, no solution will be returned: {e}", file=sys.stderr)
            return None

        tag_pose = self.config.fiducial.tag_layout[observed_tag.id]
        camera_pose = tag_pose.transformBy(
            Transform3d(from_opencv_translation(tvecs[0]), from_opencv_rotation(rvecs[0])).inverse())
        camera_pose_alt = tag_pose.transformBy(
            Transform3d(from_opencv_translation(tvecs[0]), from_opencv_rotation(rvecs[0])).inverse())

        return PoseEstimatorResult(camera_pose, reproj_errors[0][0], camera_pose_alt, reproj_errors[1][0])

    def solve_multi_target(self, visible_tags: Sequence[FiducialTagDetection]) -> Union[PoseEstimatorResult, None]:
        if (not self.config.has_tag_layout()
                or not self.config.has_calibration()
                or len(self.config.fiducial.tag_layout) == 0
                or len(visible_tags) == 0):
            return None

        tag_ids = []
        object_points = []
        image_points = []
        for tag in visible_tags:
            if tag.id not in self.config.fiducial.tag_layout or len(tag.corners) != 4:
                continue

            tag_ids.append(tag.id)
            object_points += self.__get_multi_tag_object_points(tag)
            image_points += [[tag.corners[0][0], tag.corners[0][1]],
                             [tag.corners[1][0], tag.corners[1][1]],
                             [tag.corners[2][0], tag.corners[2][1]],
                             [tag.corners[3][0], tag.corners[3][1]]]

        if len(object_points) == 0:
            return None

        try:
            retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(np.array(object_points),
                                                                      np.array(image_points),
                                                                      self.config.calibration.intrinsics_matrix,
                                                                      self.config.calibration.distortion_coefficients,
                                                                      flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error as e:
            print(f"Error in SOLVEPNP_SQPNP, no solution will be returned: {e}", file=sys.stderr)
            return None

        camera_pose = Pose3d().transformBy(Transform3d(from_opencv_translation(tvecs[0]),
                                                       from_opencv_rotation(rvecs[0])).inverse())

        return PoseEstimatorResult(camera_pose, reproj_errors[0][0], None, None)

    def __get_multi_tag_object_points(self, observed_tag: FiducialTagDetection) -> List[npt.NDArray[np.float64]]:
        tag_pose = self.config.fiducial.tag_layout[observed_tag.id]
        tag_size = self.config.fiducial.tag_size_m
        corners = [tag_pose.transformBy(Transform3d(0, tag_size / 2.0, -tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, -tag_size / 2.0, -tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, -tag_size / 2.0, tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, tag_size / 2.0, tag_size / 2.0, Rotation3d()))]
        return [to_opencv_translation(corner.translation()) for corner in corners]
