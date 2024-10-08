import logging
import time

import ntcore

from .calibration import CalibrationController, CalibrationPipeline
from .config import Config
from .output import NTOutputPublisher, StreamServer
from .pipeline import DefaultCapture, GStreamerCapture, Pipeline

logger = logging.getLogger(__name__)

NETWORK_CONFIG_FILE = 'device-config/network-config.json'
CALIBRATION_FILE = 'device-config/calibration.json'


def run_pipeline():
    logging.basicConfig(level=logging.DEBUG)

    config = Config(NETWORK_CONFIG_FILE, CALIBRATION_FILE)
    config.refresh_local()

    logger.info(
        f"Starting NT client for device {config.network.device_id}, server is set to {config.network.server_ip}")
    ntcore.NetworkTableInstance.getDefault().startClient4(config.network.device_id)
    ntcore.NetworkTableInstance.getDefault().setServer(config.network.server_ip)

    config.refresh_nt()

    capture = GStreamerCapture(config)
    pipeline = Pipeline(config)
    output = NTOutputPublisher(config)
    stream = StreamServer(config)

    calib_control = CalibrationController(config)
    calib_pipeline = CalibrationPipeline(calib_control)
    was_calibrating = False

    last_fps_time = time.perf_counter_ns()
    fps = 0
    frame_count = 0
    heartbeat = 0

    logger.info("Starting pipeline...")
    stream.start()
    while True:
        config.refresh_nt()

        ret, frame = capture.get_frame()
        if not ret:
            time.sleep(0.2)
            continue

        heartbeat += 1
        frame_count += 1
        current_time = time.perf_counter_ns()
        if current_time - last_fps_time > 1e9:
            fps = frame_count
            last_fps_time = current_time
            frame_count = 0

        result = None
        if calib_control.is_calibrating():
            if not was_calibrating:
                logger.info("Starting calibration pipeline...")
            calib_pipeline.process_frame(frame)
            was_calibrating = True
        elif was_calibrating:
            logger.info("Finishing calibration...")
            calib_pipeline.finish(CALIBRATION_FILE)
            config.refresh_local()
            was_calibrating = False
        else:
            result = pipeline.process_frame(frame)

        output.publish(result, fps, heartbeat)
        stream.set_frame(frame)
