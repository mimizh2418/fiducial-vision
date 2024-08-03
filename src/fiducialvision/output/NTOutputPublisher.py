import ntcore
from wpimath.geometry import Pose3d, Transform3d

from ..config import Config
from ..pipeline import PipelineResult


class NTOutputPublisher:
    _config: Config

    _nt_initialized: bool = False

    _timestamp_pub: ntcore.DoublePublisher
    _fps_pub: ntcore.DoublePublisher
    _heartbeat_pub: ntcore.IntegerPublisher

    _tag_ids_pub: ntcore.IntegerArrayPublisher
    _has_pose_0_pub: ntcore.BooleanPublisher
    _has_pose_1_pub: ntcore.BooleanPublisher
    _pose_0_pub: ntcore.StructPublisher
    _reproj_0_pub: ntcore.DoublePublisher
    _pose_1_pub: ntcore.StructPublisher
    _reproj_1_pub: ntcore.DoublePublisher
    _camera_to_target_0_pub: ntcore.StructArrayPublisher
    _camera_to_target_1_pub: ntcore.StructArrayPublisher

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

        tag_ids = [detection.id for detection in result.detector_result]
        self._tag_ids_pub.set(tag_ids)

        if result.pose_estimate is not None:
            self._has_pose_0_pub.set(True)
            self._pose_0_pub.set(result.pose_estimate.pose)
            self._reproj_0_pub.set(result.pose_estimate.reproj_error)
            camera_to_target_0 = [self._config.fiducial.tag_layout[tag_id] - result.pose_estimate.pose
                                  for tag_id in tag_ids]
            self._camera_to_target_0_pub.set(camera_to_target_0)
        else:
            self._has_pose_0_pub.set(False)
            self._pose_0_pub.set(Pose3d())
            self._reproj_0_pub.set(0)
            self._camera_to_target_0_pub.set([])

        if result.pose_estimate is not None and result.pose_estimate.pose_alternate is not None:
            self._has_pose_1_pub.set(True)
            self._pose_1_pub.set(result.pose_estimate.pose_alternate)
            self._reproj_1_pub.set(result.pose_estimate.reproj_error_alternate)
            camera_to_target_1 = [self._config.fiducial.tag_layout[tag_id] - result.pose_estimate.pose_alternate
                                  for tag_id in tag_ids]
            self._camera_to_target_1_pub.set(camera_to_target_1)
        else:
            self._has_pose_1_pub.set(False)
            self._pose_1_pub.set(Pose3d())
            self._reproj_1_pub.set(0)
            self._camera_to_target_1_pub.set([])

    def _init_nt(self):
        table = ntcore.NetworkTableInstance.getDefault().getTable(f"vision/{self._config.network.device_id}/output")
        pubsub_options = ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True)
        self._timestamp_pub = table.getDoubleTopic("timestamp_ns").publish(pubsub_options)
        self._fps_pub = table.getDoubleTopic("fps").publish(pubsub_options)
        self._heartbeat_pub = table.getIntegerTopic("heartbeat").publish(pubsub_options)

        self._tag_ids_pub = table.getIntegerArrayTopic("tag_ids").publish(pubsub_options)
        self._has_pose_0_pub = table.getBooleanTopic("has_pose_0").publish(pubsub_options)
        self._has_pose_1_pub = table.getBooleanTopic("has_pose_1").publish(pubsub_options)
        self._pose_0_pub = table.getStructTopic("pose_0", Pose3d).publish(pubsub_options)
        self._reproj_0_pub = table.getDoubleTopic("reproj_0").publish(pubsub_options)
        self._pose_1_pub = table.getStructTopic("pose_1", Pose3d).publish(pubsub_options)
        self._reproj_1_pub = table.getDoubleTopic("reproj_1").publish(pubsub_options)
        self._camera_to_target_0_pub = (
            table.getStructArrayTopic("camera_to_target_0", Transform3d).publish(pubsub_options))
        self._camera_to_target_1_pub = (
            table.getStructArrayTopic("camera_to_target_1", Transform3d).publish(pubsub_options))

        self._nt_initialized = True
