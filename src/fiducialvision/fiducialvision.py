import time

import cv2
import ntcore

from .config import *
from .pipeline import *


NETWORK_CONFIG_FILE = 'network_config.json'
CALIBRATION_FILE = 'calibration.json'


def run_pipeline():

    config = Config()
    config.refresh_local(NETWORK_CONFIG_FILE, CALIBRATION_FILE)

    ntcore.NetworkTableInstance.getDefault().startClient4(config.network.device_id)
    ntcore.NetworkTableInstance.getDefault().setServer(config.network.server_ip)

    config.refresh_nt()

    capture = DefaultCapture(config)
    pipeline = Pipeline(config)

    last_fps_time = time.perf_counter_ns()
    frame_count = 0

    while True:
        config.refresh_nt()

        ret, frame = capture.get_frame()
        result = pipeline.process_frame(frame)

        frame_count += 1
        current_time = time.perf_counter_ns()
        if current_time - last_fps_time > 1e9:
            fps = frame_count / ((current_time - last_fps_time) * 1e-9)
            last_fps_time = current_time
            frame_count = 0

        if result.pose_estimate is not None:
            print(result.pose_estimate.pose)
        cv2.imshow('frame', result.processed_image)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
