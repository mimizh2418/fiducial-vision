import sys
from typing import Sequence, Union

import cv2
import numpy as np
from wpimath.geometry import Pose3d, Rotation3d, Transform3d

from ..config import CameraConfig, FiducialConfig
from ..coordinate_util import from_opencv_pose, to_opencv_translation
from .pipeline_types import FiducialTagObservation, PoseEstimate


class PoseEstimator:
    def __init__(self, camera_config: CameraConfig, fiducial_config: FiducialConfig):
        self.camera_matrix = camera_config.intrinsics_matrix
        self.dist_coeffs = camera_config.distortion_coefficients
        self.tag_size = fiducial_config.tag_size_m
        self.tag_layout = fiducial_config.tag_layout

    def solve_single_target(self, observed_tag: FiducialTagObservation) -> Union[PoseEstimate, None]:
        if (len(self.tag_layout.keys()) == 0
                or observed_tag.id not in self.tag_layout.keys()
                or len(observed_tag.corners) != 4):
            return None

        object_points = np.array([[-self.tag_size / 2.0, self.tag_size / 2.0, 0],
                                  [self.tag_size / 2.0, self.tag_size / 2.0, 0],
                                  [self.tag_size / 2.0, -self.tag_size / 2.0, 0],
                                  [-self.tag_size / 2.0, -self.tag_size / 2.0, 0]])
        image_points = np.array(self.__get_image_points(observed_tag))

        try:
            retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(object_points,
                                                                      image_points,
                                                                      self.camera_matrix,
                                                                      self.dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error as e:
            print(f"Error in SOLVE_PNP_IPPE_SQUARE, no solution will be returned: {e}", file=sys.stderr)
            return None

        camera_to_tag = from_opencv_pose(rvecs[0], tvecs[0])
        camera_to_tag_alt = from_opencv_pose(rvecs[1], tvecs[1])
        world_to_camera = self.tag_layout[observed_tag.id].transformBy(
            Transform3d(camera_to_tag.translation(), camera_to_tag.rotation()).inverse())
        world_to_camera_alt = self.tag_layout[observed_tag.id].transformBy(
            Transform3d(camera_to_tag_alt.translation(), camera_to_tag_alt.rotation()).inverse())

        return PoseEstimate([observed_tag.id],
                            world_to_camera,
                            reproj_errors[0][0],
                            world_to_camera_alt,
                            reproj_errors[1][0])

    def solve_multi_target(self, visible_tags: Sequence[FiducialTagObservation]) -> Union[PoseEstimate, None]:
        if len(self.tag_layout.keys()) == 0 or len(visible_tags) == 0:
            return None

        object_points = []
        image_points = []
        for tag in visible_tags:
            if tag.id not in self.tag_layout.keys() or len(tag.corners) != 4:
                continue

            object_points += self.__get_multi_tag_object_points(tag)
            image_points += self.__get_image_points(tag)

        if len(object_points) == 0:
            return None

        try:
            retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(np.array(object_points),
                                                                      np.array(image_points),
                                                                      self.camera_matrix,
                                                                      self.dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error as e:
            print(f"Error in SOLVE_PNP_SQPNP, no solution will be returned: {e}", file=sys.stderr)
            return None

        camera_to_world = from_opencv_pose(rvecs[0], tvecs[0])
        world_to_camera = Pose3d().transformBy(
            Transform3d(camera_to_world.translation(), camera_to_world.rotation()).inverse())

        return PoseEstimate([tag.id for tag in visible_tags],
                            world_to_camera,
                            reproj_errors[0][0],
                            None,
                            None)

    def __get_multi_tag_object_points(self, observed_tag: FiducialTagObservation):
        tag_pose = self.tag_layout[observed_tag.id]
        corners = [tag_pose.transformBy(Transform3d(0, self.tag_size / 2.0, -self.tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, -self.tag_size / 2.0, -self.tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, -self.tag_size / 2.0, self.tag_size / 2.0, Rotation3d())),
                   tag_pose.transformBy(Transform3d(0, self.tag_size / 2.0, self.tag_size / 2.0, Rotation3d()))]
        return [to_opencv_translation(corner.translation()) for corner in corners]

    def __get_image_points(self, observed_tag: FiducialTagObservation):
        return [[observed_tag.corners[0][0], observed_tag.corners[0][1]],
                [observed_tag.corners[1][0], observed_tag.corners[1][1]],
                [observed_tag.corners[2][0], observed_tag.corners[2][1]],
                [observed_tag.corners[3][0], observed_tag.corners[3][1]]]
