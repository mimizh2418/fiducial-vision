import time
from typing import Tuple

import cv2

from ..config import CameraConfig
from .pipeline_types import CaptureFrame
from ..config.config_types import CameraCalibrationParams


class Capture:
    def get_frame(self) -> Tuple[bool, CaptureFrame]:
        raise NotImplementedError


class DefaultCapture(Capture):
    def __init__(self, camera_config: CameraConfig, camera_calibration: CameraCalibrationParams):
        self.config = camera_config
        self.calibration = camera_calibration
        self.video = cv2.VideoCapture(camera_config.id)
        print(self.video.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.resolution_width))
        print(self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.resolution_height))

        if (camera_config.resolution_height != camera_calibration.resolution_height
                or camera_config.resolution_width != camera_calibration.resolution_width):
            print("Warning: camera resolution does not match calibration resolution, pose estimation may be incorrect")

    def get_frame(self) -> Tuple[bool, CaptureFrame]:
        timestamp = time.time_ns()
        ret, frame = self.video.read()
        return ret, CaptureFrame(frame,
                                 timestamp,
                                 self.config.resolution_height,
                                 self.config.resolution_width,
                                 self.calibration.intrinsics_matrix,
                                 self.calibration.distortion_coefficients)

    def __del__(self):
        self.video.release()
