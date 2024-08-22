import datetime
import logging
import os.path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import cv2

from .CalibrationController import CalibrationController
from ..pipeline import CaptureFrame

logger = logging.getLogger(__name__)


class CalibrationPipeline:
    _controller: CalibrationController

    _detector: cv2.aruco.CharucoDetector
    _charuco_board: cv2.aruco.CharucoBoard
    _charuco_corners: List[npt.NDArray] = []
    _charuco_ids: List[npt.NDArray] = []
    _object_pts: List[npt.NDArray]
    _image_pts: List[npt.NDArray]
    _image_size: Tuple[int, int]

    def __init__(self, controller: CalibrationController):
        self._controller = controller
        marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        charuco_params = cv2.aruco.CharucoParameters()
        detector_params = cv2.aruco.DetectorParameters()
        self._charuco_board = cv2.aruco.CharucoBoard([12, 9], 0.030, 0.023, marker_dict)
        self._detector = cv2.aruco.CharucoDetector(self._charuco_board, charuco_params, detector_params)

    def process_frame(self, frame: CaptureFrame):
        if not self._controller.is_calibrating():
            pass

        corners, ids, _, _, = self._detector.detectBoard(frame.image)
        cv2.aruco.drawDetectedCornersCharuco(frame.image, corners, ids)
        if self._controller.should_capture_frame():
            if len(corners) < 4:
                logger.warning("Not enough ChArUco corners detected, not saving calibration frame")
                return

            object_pts, image_pts = self._charuco_board.matchImagePoints(corners, ids)
            if len(object_pts) == 0 or len(image_pts) == 0:
                logger.warning("Point matching failed, not saving calibration frame")
            self._charuco_corners += corners
            self._charuco_ids += ids
            self._object_pts += object_pts
            self._image_pts += image_pts
            self._image_size = (frame.image.shape[0], frame.image.shape[1])

            logger.info("Calibration frame saved")

    def finish(self, calibration_file: str):
        if len(self._charuco_corners) == 0:
            logger.error("No calibration data")
        elif (num_frames := len(self._charuco_corners)) < 10:
            logger.warning(
                f"Small calibration sample size {num_frames}, 10 or more frames recommended for accurate results")

        if os.path.exists(calibration_file):
            os.remove(calibration_file)

        retval, camera_mat, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(self._object_pts,
                                                                            self._image_pts,
                                                                            self._image_size,
                                                                            np.array([]),
                                                                            np.array([]))
        if not retval:
            logger.error("Calibration failed!")
            return

        avg_reproj_err = 0
        for i in range(len(self._object_pts)):
            image_pts2, _ = cv2.projectPoints(self._object_pts[i], rvecs[i], tvecs[i], camera_mat, dist_coeffs)
            avg_reproj_err += cv2.norm(self._image_pts[i], image_pts2, cv2.NORM_L2) / len(image_pts2)
        avg_reproj_err /= len(self._object_pts)

        if avg_reproj_err >= 1:
            logger.warning(f"High mean reprojection error {avg_reproj_err}, calibration may be inaccurate")

        calib_file = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_WRITE)
        calib_file.write("calibration_time", str(datetime.datetime.now()))
        calib_file.write("avg_reprojection_error", avg_reproj_err)
        calib_file.write("camera_matrix", camera_mat)
        calib_file.write("distortion_coefficients", dist_coeffs)

        logger.info(f"Calibration successful. Data saved to {calibration_file}")
