import logging
import time

import cv2

from . import PoseEstimator
from .FiducialDetector import ArUcoFiducialDetector, FiducialDetector
from .pipeline_types import CaptureFrame, PipelineResult
from ..config import Config

logger = logging.getLogger(__name__)


class Pipeline:
    _config: Config
    _fiducial_detector: FiducialDetector
    _pose_estimator: PoseEstimator

    def __init__(self, config: Config):
        self._config = config
        self._fiducial_detector = ArUcoFiducialDetector(config)
        self._pose_estimator = PoseEstimator(config)
        if not self._config.has_calibration():
            logger.warning("No calibration provided, tag transforms will not be calculated")
        if not self._config.has_tag_layout():
            logger.warning("No tag layout provided, pose estimation will not be performed")

    def process_frame(self, frame: CaptureFrame) -> PipelineResult:
        process_dt_nanos = time.perf_counter_ns()
        raw_corners, raw_ids, detections = self._fiducial_detector.detect_fiducials(frame)
        image = cv2.aruco.drawDetectedMarkers(frame.image, raw_ids, raw_corners)

        tracked_targets = []
        pose_result = None
        if self._config.has_calibration():
            if self._config.has_tag_layout():
                pose_result, tracked_targets = self._pose_estimator.solve_camera_pose(detections)
            else:
                tracked_targets = self._pose_estimator.solve_target_poses(detections)

        return PipelineResult(frame.timestamp_ns,
                              process_dt_nanos,
                              image,
                              [detection.id for detection in detections],
                              tracked_targets,
                              pose_result)
