import time

import cv2

from .config import *
from .pipeline import *


def run_pipeline():
    camera_config = load_camera_config('camera_config.json')
    fiducial_config = load_fiducial_config('fiducial_config.json')

    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.camera_resolution_width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.camera_resolution_height)
    detector = ArUcoFiducialDetector(fiducial_config)
    pose_estimator = PoseEstimator(camera_config, fiducial_config)
    pipeline = Pipeline(detector, pose_estimator)

    while True:
        ret, frame = vid.read()
        result = pipeline.process_frame(CaptureFrame(frame, time.time_ns(), 0,
                                                     camera_config.camera_resolution_height,
                                                     camera_config.camera_resolution_width,
                                                     camera_config.intrinsics_matrix,
                                                     camera_config.distortion_coefficients))

        if result.pose_estimate.has_pose:
            print(result.pose_estimate.pose)
        cv2.imshow('frame', result.processed_image)
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
