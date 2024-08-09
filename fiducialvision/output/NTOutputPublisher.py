import logging

import ntcore

from ..config import Config
from ..pipeline import PipelineResult, CameraPoseEstimate, TrackedTarget

logger = logging.getLogger(__name__)


class NTOutputPublisher:
    _config: Config

    _nt_initialized: bool = False

    _timestamp_pub: ntcore.DoublePublisher
    _fps_pub: ntcore.DoublePublisher
    _heartbeat_pub: ntcore.IntegerPublisher

    _tag_ids_pub: ntcore.IntegerArrayPublisher
    _has_pose_estimate_pub: ntcore.BooleanPublisher
    _has_tracked_targets_pub: ntcore.BooleanPublisher
    _pose_estimate_pub: ntcore.StructPublisher
    _tracked_targets_pub: ntcore.StructArrayPublisher

    def __init__(self, config: Config):
        self._config = config

    def publish(self, result: PipelineResult, fps: float, heartbeat: int):
        if not self._nt_initialized:
            self._init_nt()

        time_offset = ntcore.NetworkTableInstance.getDefault().getServerTimeOffset() or 0
        corrected_timestamp = (result.capture_timestamp_ns + time_offset * 1000)
        self._timestamp_pub.set(corrected_timestamp)
        self._fps_pub.set(fps)
        self._heartbeat_pub.set(heartbeat)

        self._tag_ids_pub.set(result.seen_tag_ids)
        self._has_pose_estimate_pub.set(result.pose_estimate is not None)
        self._has_tracked_targets_pub.set(len(result.tracked_targets) > 0)
        if result.pose_estimate is not None:
            self._pose_estimate_pub.set(result.pose_estimate)
        if len(result.tracked_targets) > 0:
            self._tracked_targets_pub.set(result.tracked_targets)

    def _init_nt(self):
        logger.info("Initializing NT output publisher")
        table = ntcore.NetworkTableInstance.getDefault().getTable(f"vision/{self._config.network.device_id}/output")
        pubsub_options = ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True)
        self._timestamp_pub = table.getDoubleTopic("timestamp_ns").publish(pubsub_options)
        self._fps_pub = table.getDoubleTopic("fps").publish(pubsub_options)
        self._heartbeat_pub = table.getIntegerTopic("heartbeat").publish(pubsub_options)

        self._tag_ids_pub = table.getIntegerArrayTopic("tag_ids").publish(pubsub_options)
        self._has_pose_estimate_pub = table.getBooleanTopic("has_pose_estimate").publish(pubsub_options)
        self._has_tracked_targets_pub = table.getBooleanTopic("has_tracked_targets").publish(pubsub_options)
        self._pose_estimate_pub = table.getStructTopic("pose_estimate", CameraPoseEstimate).publish(pubsub_options)
        self._tracked_targets_pub = table.getStructArrayTopic("tracked_targets", TrackedTarget).publish(pubsub_options)

        self._nt_initialized = True
