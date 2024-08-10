import dataclasses
import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple

import cv2

from ..config import CameraConfig, Config
from .pipeline_types import CaptureFrame

logger = logging.getLogger(__name__)


class Capture(ABC):
    @abstractmethod
    def get_frame(self) -> Tuple[bool, CaptureFrame]:
        pass

    @abstractmethod
    def _update_config(self):
        pass


class DefaultCapture(Capture):
    _config: CameraConfig
    _last_config: CameraConfig
    _video: cv2.VideoCapture

    def __init__(self, config: Config):
        self._config = config.camera
        self._last_config = dataclasses.replace(self._config)
        self._video = cv2.VideoCapture(self._config.id)
        self._update_config()

    def get_frame(self) -> Tuple[bool, CaptureFrame]:
        if self._last_config != self._config:
            logger.debug("Camera configuration changed, reapplying settings")
            self._update_config()

        timestamp = time.time_ns()
        ret, frame = self._video.read()
        return ret, CaptureFrame(frame,
                                 timestamp,
                                 self._config.resolution_height,
                                 self._config.resolution_width)

    def _update_config(self):
        if self._last_config.id != self._config.id:
            self._video.open(self._config.id)
        self._video.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.resolution_width)
        self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.resolution_height)
        self._video.set(cv2.CAP_PROP_AUTO_EXPOSURE, self._config.auto_exposure)
        self._video.set(cv2.CAP_PROP_EXPOSURE, self._config.exposure)
        self._video.set(cv2.CAP_PROP_BRIGHTNESS, self._config.brightness)
        self._video.set(cv2.CAP_PROP_GAIN, self._config.gain)
        self._last_config = dataclasses.replace(self._config)

    def __del__(self):
        self._video.release()
