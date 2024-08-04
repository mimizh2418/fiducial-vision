import time

import cv2

from . import PoseEstimator
from .FiducialDetector import ArUcoFiducialDetector
from .pipeline_types import CaptureFrame, PipelineResult
from ..config import Config


class Pipeline:
    def __init__(self, config: Config):
        self.fiducial_detector = ArUcoFiducialDetector(config)
        self.pose_estimator = PoseEstimator(config) if config.has_calibration() and config.has_tag_layout() else None
        self.estimate_poses = self.pose_estimator is not None

    def process_frame(self, frame: CaptureFrame) -> PipelineResult:
        process_dt_nanos = time.perf_counter_ns()
        raw_corners, raw_ids, detections = self.fiducial_detector.detect_fiducials(frame)
        image = cv2.aruco.drawDetectedMarkers(frame.image,
                                              raw_ids,
                                              raw_corners)
        pose_result = None
        if self.estimate_poses:
            pose_result = self.pose_estimator.solve_camera_pose(detections)
        process_dt_nanos = time.perf_counter_ns() - process_dt_nanos

        return PipelineResult(frame.timestamp_ns,
                              process_dt_nanos,
                              image,
                              detections,
                              pose_result)
