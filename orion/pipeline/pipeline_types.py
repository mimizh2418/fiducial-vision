from dataclasses import dataclass, field
from typing import Sequence, Optional

import cv2
import numpy as np
from numpy import typing as npt
from wpimath.geometry import Pose3d, Transform3d
from wpiutil.wpistruct import make_wpistruct


@dataclass(frozen=True)
class CaptureFrame:
    image: cv2.Mat
    timestamp_ns: int
    resolution_height: int
    resolution_width: int


@dataclass(frozen=True)
class FiducialTagDetection:
    id: int
    corners: npt.NDArray[np.float64]


@make_wpistruct(name="TrackedTarget")
@dataclass
class TrackedTarget:
    id: int
    camera_to_target: Transform3d
    reproj_error: float
    camera_to_target_alt: Transform3d = field(default_factory=lambda: Transform3d())
    reproj_error_alt: float = 0.0
    has_alt: bool = field(init=False)

    def __post_init__(self):
        self.has_alt = self.camera_to_target_alt != Transform3d() and self.reproj_error_alt != 0.0


@make_wpistruct(name="CameraPoseEstimate")
@dataclass
class CameraPoseEstimate:
    pose: Pose3d
    reproj_error: float
    pose_alt: Pose3d = field(default_factory=lambda: Pose3d())
    reproj_error_alt: float = 0.0
    has_alt: bool = field(init=False)

    def __post_init__(self):
        self.has_alt = self.pose_alt != Pose3d() and self.reproj_error_alt != 0.0


@dataclass(frozen=True)
class PipelineResult:
    capture_timestamp_ns: int
    process_dt_ns: int
    processed_image: cv2.Mat

    seen_tag_ids: Sequence[int]
    tracked_targets: Sequence[TrackedTarget]
    pose_estimate: Optional[CameraPoseEstimate]
