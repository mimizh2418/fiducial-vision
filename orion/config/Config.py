import json
import logging
from typing import Union

import cv2
import ntcore
import numpy as np
from wpimath.geometry import Pose3d, Rotation3d, Quaternion

from .config_types import NetworkConfig, CameraConfig, Calibration, FiducialConfig

logger = logging.getLogger(__name__)


class Config:
    fiducial_families = {
        "aruco_4x4_50": cv2.aruco.DICT_4X4_50,
        "aruco_4x4_100": cv2.aruco.DICT_4X4_100,
        "aruco_4x4_250": cv2.aruco.DICT_4X4_250,
        "aruco_4x4_1000": cv2.aruco.DICT_4X4_1000,
        "aruco_5x5_50": cv2.aruco.DICT_5X5_50,
        "aruco_5x5_100": cv2.aruco.DICT_5X5_100,
        "aruco_5x5_250": cv2.aruco.DICT_5X5_250,
        "aruco_5x5_1000": cv2.aruco.DICT_5X5_1000,
        "aruco_6x6_50": cv2.aruco.DICT_6X6_50,
        "aruco_6x6_100": cv2.aruco.DICT_6X6_100,
        "aruco_6x6_250": cv2.aruco.DICT_6X6_250,
        "aruco_6x6_1000": cv2.aruco.DICT_6X6_1000,
        "aruco_7x7_50": cv2.aruco.DICT_7X7_50,
        "aruco_7x7_100": cv2.aruco.DICT_7X7_100,
        "aruco_7x7_250": cv2.aruco.DICT_7X7_250,
        "aruco_7x7_1000": cv2.aruco.DICT_7X7_1000,
        "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "apriltag_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "apriltag_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "aruco_mip_36h12": cv2.aruco.DICT_ARUCO_MIP_36h12
    }

    network: NetworkConfig
    camera: CameraConfig
    calibration: Union[Calibration, None]
    fiducial: FiducialConfig

    network_config_file: str
    calibration_file: str

    _nt_initialized: bool = False
    _camera_id_entry: ntcore.StringEntry
    _camera_resolution_w_entry: ntcore.IntegerEntry
    _camera_resolution_h_entry: ntcore.IntegerEntry
    _camera_auto_exposure_entry: ntcore.IntegerEntry
    _camera_exposure_entry: ntcore.IntegerEntry
    _camera_brightness_entry: ntcore.IntegerEntry
    _camera_gain_entry: ntcore.IntegerEntry
    _tag_family_entry: ntcore.StringEntry
    _tag_size_entry: ntcore.DoubleEntry
    _tag_layout_entry: ntcore.StringEntry

    _last_family_update: int = -1
    _last_layout_update: int = -1

    def __init__(self, network_config_file: str, calibration_file: str):
        self.network_config_file = network_config_file
        self.calibration_file = calibration_file
        self.network = NetworkConfig()
        self.camera = CameraConfig()
        self.calibration = Calibration()
        self.fiducial = FiducialConfig()

    def refresh_local(self):
        logger.info(f"Loading network config from {self.network_config_file}...")
        try:
            with open(self.network_config_file, "r") as f:
                network_data = json.loads(f.read())
                self.network.device_id = network_data["device_id"]
                self.network.server_ip = network_data["server_ip"]
                self.network.stream_port = network_data["stream_port"]
        except FileNotFoundError:
            logger.error(f"Network config file {self.network_config_file} not found, using defaults")

        calib_data = cv2.FileStorage(self.calibration_file, cv2.FILE_STORAGE_READ)
        intrinsics_mat = calib_data.getNode("camera_matrix").mat()
        dist_coeffs = calib_data.getNode("distortion_coefficients").mat()
        calib_data.release()

        if type(intrinsics_mat) is not np.ndarray or type(dist_coeffs) is not np.ndarray:
            logger.warning(f"Calibration file {self.calibration_file} not found or invalid, pose estimation disabled")
            self.calibration = None
        else:
            self.calibration.intrinsics_matrix = intrinsics_mat
            self.calibration.distortion_coeffs = dist_coeffs

    def refresh_nt(self):
        if not self._nt_initialized:
            self._init_nt()

        camera_id = self._camera_id_entry.get()
        try:
            camera_id = int(camera_id)
        except ValueError:
            pass
        self.camera.id = camera_id
        self.camera.resolution_width = self._camera_resolution_w_entry.get()
        self.camera.resolution_height = self._camera_resolution_h_entry.get()
        self.camera.auto_exposure = self._camera_auto_exposure_entry.get()
        self.camera.exposure = self._camera_exposure_entry.get()
        self.camera.brightness = self._camera_brightness_entry.get()
        self.camera.gain = self._camera_gain_entry.get()

        self.fiducial.tag_size_m = self._tag_size_entry.get()

        if (family_change := self._tag_family_entry.getLastChange()) > self._last_family_update:
            tag_family = self._tag_family_entry.get()
            if tag_family in self.fiducial_families:
                self.fiducial.tag_family = self.fiducial_families[tag_family]
                logger.debug(f"Set tag family to {tag_family}")
            else:
                logger.warning('Unknown tag family "{tag_family}", defaulting to apriltag_36h11')
                self.fiducial.tag_family = cv2.aruco.DICT_APRILTAG_36h11
            self._last_family_update = family_change

        if (layout_change := self._tag_layout_entry.getLastChange()) > self._last_layout_update:
            try:
                self.fiducial.tag_layout = {}
                tag_layout_data = json.loads(self._tag_layout_entry.get())
                for tag_data in tag_layout_data["tags"]:
                    tag_id = tag_data["ID"]
                    tag_pose = Pose3d(tag_data["pose"]["translation"]["x"],
                                      tag_data["pose"]["translation"]["y"],
                                      tag_data["pose"]["translation"]["z"],
                                      Rotation3d(Quaternion(tag_data["pose"]["rotation"]["quaternion"]["w"],
                                                            tag_data["pose"]["rotation"]["quaternion"]["x"],
                                                            tag_data["pose"]["rotation"]["quaternion"]["y"],
                                                            tag_data["pose"]["rotation"]["quaternion"]["z"])))
                    self.fiducial.tag_layout[tag_id] = tag_pose
                logger.debug("Successfully loaded tag layout")
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Failed to load tag layout, invalid format")
                self.fiducial.tag_layout = None
            self._last_layout_update = layout_change

    def _init_nt(self):
        logger.info("Initializing NetworkTables config...")

        table = ntcore.NetworkTableInstance.getDefault().getTable(f"orion/{self.network.device_id}/config")

        self._camera_id_entry = table.getStringTopic("camera_id").getEntry(str(self.camera.id))
        self._camera_resolution_w_entry = (
            table.getIntegerTopic("camera_resolution_width").getEntry(self.camera.resolution_width))
        self._camera_resolution_h_entry = (
            table.getIntegerTopic("camera_resolution_height").getEntry(self.camera.resolution_height))
        self._camera_auto_exposure_entry = (
            table.getIntegerTopic("camera_auto_exposure").getEntry(self.camera.auto_exposure))
        self._camera_exposure_entry = table.getIntegerTopic("camera_exposure").getEntry(self.camera.exposure)
        self._camera_brightness_entry = table.getIntegerTopic("camera_brightness").getEntry(self.camera.brightness)
        self._camera_gain_entry = table.getIntegerTopic("camera_gain").getEntry(self.camera.gain)
        self._tag_family_entry = table.getStringTopic("tag_family").getEntry("apriltag_36h11")
        self._tag_size_entry = table.getDoubleTopic("tag_size_m").getEntry(self.fiducial.tag_size_m)
        self._tag_layout_entry = table.getStringTopic("tag_layout").getEntry("")

        self._camera_id_entry.setDefault(str(self.camera.id))
        self._camera_resolution_w_entry.setDefault(self.camera.resolution_width)
        self._camera_resolution_h_entry.setDefault(self.camera.resolution_height)
        self._camera_auto_exposure_entry.setDefault(self.camera.auto_exposure)
        self._camera_exposure_entry.setDefault(self.camera.exposure)
        self._camera_brightness_entry.setDefault(self.camera.brightness)
        self._camera_gain_entry.setDefault(self.camera.gain)
        self._tag_family_entry.setDefault("apriltag_36h11")
        self._tag_size_entry.setDefault(self.fiducial.tag_size_m)
        self._tag_layout_entry.setDefault("")

        self._camera_id_entry.getTopic().setRetained(True)
        self._camera_resolution_w_entry.getTopic().setRetained(True)
        self._camera_resolution_h_entry.getTopic().setRetained(True)
        self._camera_auto_exposure_entry.getTopic().setRetained(True)
        self._camera_exposure_entry.getTopic().setRetained(True)
        self._camera_brightness_entry.getTopic().setRetained(True)
        self._camera_gain_entry.getTopic().setRetained(True)
        self._tag_family_entry.getTopic().setRetained(True)
        self._tag_size_entry.getTopic().setRetained(True)
        self._tag_layout_entry.getTopic().setRetained(True)

        self._nt_initialized = True

    def has_calibration(self) -> bool:
        return self.calibration is not None

    def has_tag_layout(self) -> bool:
        return self.fiducial.tag_layout is not None
