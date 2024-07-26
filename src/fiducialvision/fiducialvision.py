import time

import cv2

from .config import *
from .pipeline import *


def run_pipeline():
    camera_config = load_camera_config('camera_config.json')
    calib_params = load_camera_calibration('calibration.json')
    fiducial_config = load_fiducial_config('fiducial_config.json')

    capture = DefaultCapture(camera_config, calib_params)
    detector = ArUcoFiducialDetector(fiducial_config)
    pose_estimator = PoseEstimator(fiducial_config)
    pipeline = Pipeline(detector, pose_estimator)

    last_fps_time = time.perf_counter_ns()
    frame_count = 0

    while True:
        ret, frame = capture.get_frame()
        result = pipeline.process_frame(frame)

        frame_count += 1
        current_time = time.perf_counter_ns()
        if current_time - last_fps_time > 1e9:
            fps = frame_count / ((current_time - last_fps_time) * 1e-9)
            last_fps_time = current_time
            frame_count = 0

        if result.pose_estimate.has_pose:
            print(result.pose_estimate.pose)
        cv2.imshow('frame', result.processed_image)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
