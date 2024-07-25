import time
from typing import Union

from . import PoseEstimator
from .FiducialDetector import FiducialDetector
from .pipeline_types import CaptureFrame, PipelineResult, PoseEstimatorResult


class Pipeline:
    def __init__(self, fiducial_detector: FiducialDetector, pose_estimator: Union[None, PoseEstimator] = None):
        self.fiducial_detector = fiducial_detector
        self.pose_estimator = pose_estimator
        self.estimate_poses = pose_estimator is not None

    def process_frame(self, frame: CaptureFrame) -> PipelineResult:
        process_dt_nanos = time.perf_counter_ns()
        detector_result = self.fiducial_detector.process_frame(frame)
        pose_result = PoseEstimatorResult([], None, None, None, None)
        if self.estimate_poses:
            if len(detector_result.tag_detections) == 1:
                pose_result = self.pose_estimator.solve_single_target(detector_result.tag_detections[0])
            else:
                pose_result = self.pose_estimator.solve_multi_target(detector_result.tag_detections)
        process_dt_nanos = time.perf_counter_ns() - process_dt_nanos

        return PipelineResult(frame.timestamp_nanos,
                              process_dt_nanos,
                              detector_result.processed_image,
                              detector_result,
                              pose_result)
