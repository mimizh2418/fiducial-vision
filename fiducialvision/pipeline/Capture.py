import dataclasses
import time
from typing import Tuple

import cv2

from ..config import CameraConfig, Config
from .pipeline_types import CaptureFrame


class Capture:
    def get_frame(self) -> Tuple[bool, CaptureFrame]:
        raise NotImplementedError


class DefaultCapture(Capture):
    _config: CameraConfig
    _last_config: CameraConfig
    _video: cv2.VideoCapture

    def __init__(self, config: Config):
        self._config = config.camera
        self._last_config = dataclasses.replace(self._config)
        self._video = cv2.VideoCapture(self._config.id)
        self._video.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.resolution_width)
        self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.resolution_height)
        self._video.set(cv2.CAP_PROP_AUTO_EXPOSURE, self._config.auto_exposure)
        self._video.set(cv2.CAP_PROP_EXPOSURE, self._config.exposure)
        self._video.set(cv2.CAP_PROP_GAIN, self._config.gain)

    def get_frame(self) -> Tuple[bool, CaptureFrame]:
        if self._last_config != self._config:
            if self._last_config.id != self._config.id:
                self._video.open(self._config.id)
            self._video.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.resolution_width)
            self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.resolution_height)
            self._video.set(cv2.CAP_PROP_AUTO_EXPOSURE, self._config.auto_exposure)
            self._video.set(cv2.CAP_PROP_EXPOSURE, self._config.exposure)
            self._video.set(cv2.CAP_PROP_GAIN, self._config.gain)
            self._last_config = dataclasses.replace(self._config)

        timestamp = time.time_ns()
        ret, frame = self._video.read()
        return ret, CaptureFrame(frame,
                                 timestamp,
                                 self._last_config.resolution_height,
                                 self._last_config.resolution_width)

    def __del__(self):
        self._video.release()
