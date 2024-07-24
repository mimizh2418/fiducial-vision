import cv2

from .config import *
from .pipeline import *


def run_pipeline():
    camera_config = load_camera_config('camera_config.json')
    fiducial_config = load_fiducial_config('fiducial_config.json')

    vid = cv2.VideoCapture(camera_config.index)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.camera_resolution_width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.camera_resolution_height)
    detector = FiducialDetector(fiducial_config)
    pose_estimator = PoseEstimator(camera_config, fiducial_config)

    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        detector_results = detector.detect_fiducials(frame)
        if len(detector_results) > 0:
            print(pose_estimator.solve_single_target(detector_results[0]))
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
