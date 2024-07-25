import sys
from typing import Sequence

import cv2
import numpy as np
from wpimath.geometry import Pose3d, Rotation3d, Transform3d

from ..config import CameraConfig, FiducialConfig
from ..coordinate_util import to_opencv_translation, from_opencv_translation, from_opencv_rotation
from .pipeline_types import FiducialTagDetection, PoseEstimatorResult


class PoseEstimator:
    def __init__(self, camera_config: CameraConfig, fiducial_config: FiducialConfig):
        self.camera_matrix = camera_config.intrinsics_matrix
        self.dist_coeffs = camera_config.distortion_coefficients
        self.tag_size = fiducial_config.tag_size_m
        self.tag_layout = fiducial_config.tag_layout
        self.accepted_ids = fiducial_config.tag_layout.keys()

    def solve_single_target(self, observed_tag: FiducialTagDetection) -> PoseEstimatorResult:
        if (len(self.accepted_ids) == 0
                or observed_tag.id not in self.accepted_ids
                or len(observed_tag.corners) != 4):
            return PoseEstimatorResult([], None, None, None, None)

        object_points = np.array([[-self.tag_size / 2.0, self.tag_size / 2.0, 0],
                                  [self.tag_size / 2.0, self.tag_size / 2.0, 0],
                                  [self.tag_size / 2.0, -self.tag_size / 2.0, 0],
                                  [-self.tag_size / 2.0, -self.tag_size / 2.0, 0]])
        image_points = observed_tag.corners

        try:
            retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(object_points,
                                                                      image_points,
                                                                      self.camera_matrix,
                                                                      self.dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error as e:
            print(f"Error in SOLVE_PNP_IPPE_SQUARE, no solution will be returned: {e}", file=sys.stderr)
            return PoseEstimatorResult([], None, None, None, None)

        tag_pose = self.tag_layout[observed_tag.id]
        camera_pose = tag_pose.transformBy(
            Transform3d(from_opencv_translation(tvecs[0]), from_opencv_rotation(rvecs[0])).inverse())
        camera_pose_alt = tag_pose.transformBy(
            Transform3d(from_opencv_translation(tvecs[0]), from_opencv_rotation(rvecs[0])).inverse())

        return PoseEstimatorResult([observed_tag],
                                   camera_pose,
                                   reproj_errors[0][0],
                                   camera_pose_alt,
                                   reproj_errors[1][0])

    def solve_multi_target(self, visible_tags: Sequence[FiducialTagDetection]) -> PoseEstimatorResult:
        if len(self.accepted_ids) == 0 or len(visible_tags) == 0:
            return PoseEstimatorResult([], None, None, None, None)

        tag_ids = []
        object_points = []
        image_points = []
        for tag in visible_tags:
            if tag.id not in self.accepted_ids or len(tag.corners) != 4:
                continue

            tag_ids.append(tag.id)
            object_points += self.__get_multi_tag_object_points(tag)
            image_points += [[tag.corners[0][0], tag.corners[0][1]],
                             [tag.corners[1][0], tag.corners[1][1]],
                             [tag.corners[2][0], tag.corners[2][1]],
                             [tag.corners[3][0], tag.corners[3][1]]]

        if len(object_points) == 0:
            return PoseEstimatorResult([], None, None, None, None)

        try:
            retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(np.array(object_points),
                                                                      np.array(image_points),
                                                                      self.camera_matrix,
                                                                      self.dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error as e:
            print(f"Error in SOLVE_PNP_SQPNP, no solution will be returned: {e}", file=sys.stderr)
            return PoseEstimatorResult([], None, None, None, None)

        camera_pose = Pose3d().transformBy(Transform3d(from_opencv_translation(tvecs[0]),
                                                       from_opencv_rotation(rvecs[0])).inverse())

        return PoseEstimatorResult(visible_tags, camera_pose, reproj_errors[0][0], None, None)

    def __get_multi_tag_object_points(self, observed_tag: FiducialTagDetection):
        tag_pose = self.tag_layout[observed_tag.id]
        corners = [tag_pose.transformBy(Transform3d(0, self.tag_size / 2.0, -self.tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, -self.tag_size / 2.0, -self.tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, -self.tag_size / 2.0, self.tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, self.tag_size / 2.0, self.tag_size / 2.0, Rotation3d()))]
        return [to_opencv_translation(corner.translation()) for corner in corners]
