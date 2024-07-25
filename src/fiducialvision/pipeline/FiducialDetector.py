import math
import sys
from typing import Tuple, Sequence

import cv2
import numpy as np
import numpy.typing as npt

from ..config import FiducialConfig
from .pipeline_types import FiducialTagDetection, CaptureFrame, FiducialDetectorResult


class FiducialDetector:
    def __init__(self, fiducial_config: FiducialConfig):
        self.accepted_ids = fiducial_config.tag_layout.keys()

    def detect_fiducials(self, frame: CaptureFrame) -> Tuple[Sequence[npt.NDArray[np.float64]], npt.NDArray[np.int32]]:
        """
        :param frame: CaptureFrame object containing image to detect fiducials in
        :return: tuple consisting of array detected fiducial corners and array of corresponding ids
        """
        raise NotImplementedError

    def process_frame(self, frame: CaptureFrame) -> FiducialDetectorResult:
        corners, ids = self.detect_fiducials(frame)
        processed_image = cv2.aruco.drawDetectedMarkers(frame.image, corners, ids)
        if len(corners) == 0:
            return FiducialDetectorResult(processed_image, [])

        detections = []
        for tag_id, corner_pts in zip(ids, corners):
            if tag_id[0] not in self.accepted_ids:
                continue

            x_sum = corner_pts[0][0][0] + corner_pts[0][1][0] + corner_pts[0][2][0] + corner_pts[0][3][0]
            y_sum = corner_pts[0][0][1] + corner_pts[0][1][1] + corner_pts[0][2][1] + corner_pts[0][3][1]
            center_pt = np.array([x_sum / 4, y_sum / 4])

            # Compute undistorted center point for yaw and pitch estimation
            # TODO: maybe loosen termination criteria
            term_criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 30, 1e-5)
            center_pt_undist = cv2.undistortImagePoints(np.array([center_pt]),
                                                        frame.camera_intrinsics_mat,
                                                        frame.camera_dist_coeffs,
                                                        arg1=term_criteria)[0][0]

            if (math.isnan(center_pt_undist[0])
                    or math.isinf(center_pt_undist[0])
                    or math.isnan(center_pt_undist[1])
                    or math.isinf(center_pt_undist[1])):
                print("undistortImagePoints failed to find a solution, original point will be used", file=sys.stderr)
                center_pt_undist = center_pt

            # Compute yaw and pitch
            focal_length_x = frame.camera_intrinsics_mat[0][0]
            focal_length_y = frame.camera_intrinsics_mat[1][1]
            center_x = frame.camera_intrinsics_mat[0][2]
            center_y = frame.camera_intrinsics_mat[1][2]
            yaw = math.atan((center_x - center_pt_undist[0]) / focal_length_x)
            pitch = math.atan((center_y - center_pt_undist[1]) / focal_length_y)

            detections.append(FiducialTagDetection(tag_id[0], corner_pts[0], center_pt, yaw, pitch, 0))

        return FiducialDetectorResult(processed_image, detections)


class ArUcoFiducialDetector(FiducialDetector):
    def __init__(self, fiducial_config: FiducialConfig):
        super().__init__(fiducial_config)
        detector_params = cv2.aruco.DetectorParameters()
        marker_dict = cv2.aruco.getPredefinedDictionary(fiducial_config.tag_family)
        self.detector = cv2.aruco.ArucoDetector(marker_dict, detector_params)

    def detect_fiducials(self, frame: CaptureFrame) -> Tuple[Sequence[npt.NDArray[np.float64]], npt.NDArray[np.int32]]:
        corners, ids, rejected_points = self.detector.detectMarkers(frame.image)
        return corners, ids
