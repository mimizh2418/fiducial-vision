from typing import List

import cv2

from ..config import FiducialConfig
from .pipeline_types import FiducialTagObservation


class FiducialDetector:
    def __init__(self, fiducial_config: FiducialConfig):
        detector_params = cv2.aruco.DetectorParameters()
        marker_dict = cv2.aruco.getPredefinedDictionary(fiducial_config.tag_family)
        self.accepted_ids = fiducial_config.tag_layout.keys()
        self.detector = cv2.aruco.ArucoDetector(marker_dict, detector_params)

    def detect_fiducials(self, image: cv2.Mat) -> List[FiducialTagObservation]:
        corners, ids, rejected_points = self.detector.detectMarkers(image)
        if len(corners) == 0:
            return []
        return [FiducialTagObservation(tag_id[0], corner_pts[0])
                for tag_id, corner_pts in zip(ids, corners)
                if tag_id[0] in self.accepted_ids]
