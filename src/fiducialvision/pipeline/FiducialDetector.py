from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from ..config import Config
from .pipeline_types import FiducialTagDetection, CaptureFrame


class FiducialDetector:
    _config: Config

    def __init__(self, config: Config):
        self._config = config

    def detect_fiducials(self, frame: CaptureFrame) -> tuple[npt.NDArray[np.int32],
                                                             Sequence[npt.NDArray[np.float64]],
                                                             Sequence[FiducialTagDetection]]:
        raise NotImplementedError


class ArUcoFiducialDetector(FiducialDetector):
    _detector: cv2.aruco.ArucoDetector

    def __init__(self, config: Config):
        super().__init__(config)
        detector_params = cv2.aruco.DetectorParameters()
        marker_dict = cv2.aruco.getPredefinedDictionary(config.fiducial.tag_family)
        self._detector = cv2.aruco.ArucoDetector(marker_dict, detector_params)

    def detect_fiducials(self, frame: CaptureFrame) -> tuple[npt.NDArray[np.int32],
                                                             Sequence[npt.NDArray[np.float64]],
                                                             Sequence[FiducialTagDetection]]:
        corners, ids, rejected_pts = self._detector.detectMarkers(frame.image)
        if len(corners) == 0:
            return ids, corners, []

        detections = [FiducialTagDetection(tag_id[0], corner_pts[0])
                      for tag_id, corner_pts in zip(ids, corners)
                      if self._config.has_tag_layout() and tag_id[0] in self._config.fiducial.tag_layout]
        return ids, corners, detections
