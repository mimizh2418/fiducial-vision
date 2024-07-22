from typing import List

from ..config import CameraConfig, FiducialConfig
from .pipeline_types import FiducialTagObservation, PoseEstimate


class PoseEstimator:
    def __init__(self, camera_config: CameraConfig, fiducial_config: FiducialConfig):
        self.camera_matrix = camera_config.intrinsics_matrix
        self.dist_coeffs = camera_config.distortion_coefficients
        self.tag_size = fiducial_config.tag_size_m
        self.tag_layout = fiducial_config.tag_layout

    def solve_single_target(self, observed_tag: FiducialTagObservation) -> PoseEstimate:
        # TODO implement
        pass

    def solve_multi_target(self, visible_tags: List[FiducialTagObservation]) -> PoseEstimate:
        # TODO implement
        pass
