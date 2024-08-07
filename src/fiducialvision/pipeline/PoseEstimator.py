import sys
from typing import Sequence, Optional

import cv2
import numpy as np
import numpy.typing as npt
from wpimath.geometry import Pose3d, Rotation3d, Transform3d

from ..config import Config
from ..coordinate_util import to_opencv_translation, from_opencv_translation, from_opencv_rotation
from .pipeline_types import FiducialTagDetection, CameraPoseEstimate, TrackedTarget


class PoseEstimator:
    config: Config

    def __init__(self, config: Config):
        self.config = config

    def solve_camera_pose(self, observed_tags: Sequence[FiducialTagDetection]) -> tuple[Optional[CameraPoseEstimate],
                                                                                        Sequence[TrackedTarget]]:
        if (not self.config.has_tag_layout()
                or not self.config.has_calibration()
                or len(self.config.fiducial.tag_layout) == 0
                or len(observed_tags) == 0):
            return None, []

        if len(observed_tags) == 1:
            # Single tag visible, use IPPE_SQUARE
            if observed_tags[0].id not in self.config.fiducial.tag_layout:
                return None, []

            object_points = self._get_single_tag_object_pts()
            image_points = observed_tags[0].corners

            try:
                retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(object_points,
                                                                          image_points,
                                                                          self.config.calibration.intrinsics_matrix,
                                                                          self.config.calibration.distortion_coeffs,
                                                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except cv2.error as e:
                print(f"Error in SOLVEPNP_IPPE_SQUARE, no solution will be returned: {e}", file=sys.stderr)
                return None, []

            tag_pose = self.config.fiducial.tag_layout[observed_tags[0].id]
            camera_pose = tag_pose.transformBy(
                Transform3d(from_opencv_translation(tvecs[0]), from_opencv_rotation(rvecs[0])).inverse())
            camera_pose_alt = tag_pose.transformBy(
                Transform3d(from_opencv_translation(tvecs[0]), from_opencv_rotation(rvecs[0])).inverse())

            return (CameraPoseEstimate(camera_pose, reproj_errors[0][0], camera_pose_alt, reproj_errors[1][0]),
                    [TrackedTarget(observed_tags[0].id,
                                   tag_pose - camera_pose,
                                   reproj_errors[0][0],
                                   tag_pose - camera_pose_alt,
                                   reproj_errors[1][0])])
        else:
            # Do multi-tag estimation
            object_points = []
            image_points = []
            for tag in observed_tags:
                if tag.id not in self.config.fiducial.tag_layout or len(tag.corners) != 4:
                    continue

                object_points += self._get_multi_tag_object_pts(tag)
                image_points += [[tag.corners[0][0], tag.corners[0][1]],
                                 [tag.corners[1][0], tag.corners[1][1]],
                                 [tag.corners[2][0], tag.corners[2][1]],
                                 [tag.corners[3][0], tag.corners[3][1]]]

            if len(object_points) == 0:
                return None, []

            try:
                retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(np.array(object_points),
                                                                          np.array(image_points),
                                                                          self.config.calibration.intrinsics_matrix,
                                                                          self.config.calibration.distortion_coeffs,
                                                                          flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error as e:
                print(f"Error in SOLVEPNP_SQPNP, no solution will be returned: {e}", file=sys.stderr)
                return None, []

            camera_pose = Pose3d().transformBy(Transform3d(from_opencv_translation(tvecs[0]),
                                                           from_opencv_rotation(rvecs[0])).inverse())
            return (CameraPoseEstimate(camera_pose, reproj_errors[0][0]),
                    [TrackedTarget(tag.id, self.config.fiducial.tag_layout[tag.id] - camera_pose, reproj_errors[0][0])
                     for tag in observed_tags])

    def solve_target_poses(self, observed_tags: Sequence[FiducialTagDetection]) -> Sequence[TrackedTarget]:
        if not self.config.has_calibration or len(observed_tags) == 0:
            return []

        tracked_targets = []
        object_points = self._get_single_tag_object_pts()
        for tag in observed_tags:
            if self.config.has_tag_layout() and tag.id not in self.config.fiducial.tag_layout:
                continue
            try:
                retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(object_points,
                                                                          tag.corners,
                                                                          self.config.calibration.intrinsics_matrix,
                                                                          self.config.calibration.distortion_coeffs,
                                                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except cv2.error as e:
                print(f"Error in SOLVEPNP_IPPE_SQUARE, could not compute pose for tag {tag.id}: {e}", file=sys.stderr)
                continue
            tracked_targets.append(TrackedTarget(tag.id,
                                                 Transform3d(from_opencv_translation(tvecs[0]),
                                                             from_opencv_rotation(rvecs[0])),
                                                 reproj_errors[0][0],
                                                 Transform3d(from_opencv_translation(tvecs[1]),
                                                             from_opencv_rotation(rvecs[1])),
                                                 reproj_errors[1][0]))
        return tracked_targets

    def _get_single_tag_object_pts(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.config.fiducial.tag_size_m / 2.0, self.config.fiducial.tag_size_m / 2.0, 0],
                         [self.config.fiducial.tag_size_m / 2.0, self.config.fiducial.tag_size_m / 2.0, 0],
                         [self.config.fiducial.tag_size_m / 2.0, -self.config.fiducial.tag_size_m / 2.0, 0],
                         [-self.config.fiducial.tag_size_m / 2.0, -self.config.fiducial.tag_size_m / 2.0, 0]])

    def _get_multi_tag_object_pts(self, observed_tag: FiducialTagDetection) -> Sequence[npt.NDArray[np.float64]]:
        tag_pose = self.config.fiducial.tag_layout[observed_tag.id]
        tag_size = self.config.fiducial.tag_size_m
        corners = [tag_pose.transformBy(Transform3d(0, tag_size / 2.0, -tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, -tag_size / 2.0, -tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, -tag_size / 2.0, tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, tag_size / 2.0, tag_size / 2.0, Rotation3d()))]
        return [to_opencv_translation(corner.translation()) for corner in corners]
