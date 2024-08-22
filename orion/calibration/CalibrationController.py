import ntcore

from ..config import Config


class CalibrationController:
    _config: Config

    _nt_initialized: bool = False
    _is_calibrating_entry: ntcore.BooleanEntry
    _capture_frame_entry: ntcore.BooleanEntry

    def __init__(self, config: Config):
        self._config = config

    def is_calibrating(self) -> bool:
        if not self._nt_initialized:
            self._init_nt()
        calibrating = self._is_calibrating_entry.get()
        if not calibrating:
            self._capture_frame_entry.set(False)
        return calibrating

    def should_capture_frame(self) -> bool:
        if not self._nt_initialized:
            self._init_nt()
        if self._capture_frame_entry.get():
            self._capture_frame_entry.set(False)
            return True
        return False

    def _init_nt(self):
        table = ntcore.NetworkTableInstance.getDefault().getTable(f"orion/{self._config.network.device_id}/calibration")
        self._is_calibrating_entry = table.getBooleanTopic("is_calibrating").getEntry(False)
        self._capture_frame_entry = table.getBooleanTopic("capture_frame").getEntry(False)
        self._is_calibrating_entry.set(False)
        self._capture_frame_entry.set(False)
        self._nt_initialized = True
