import cv2

from .config import *
from .pipeline import *


def run_pipeline():
    camera_config = load_camera_config('camera_config.json')
    fiducial_config = load_fiducial_config('fiducial_config.json')

    vid = cv2.VideoCapture(camera_config.index)
    detector = FiducialDetector(fiducial_config)
    pose_estimator = PoseEstimator(camera_config, fiducial_config)

    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        detector_results = detector.detect_fiducials(frame)
        if len(detector_results) > 0:
            print(pose_estimator.solve_multi_target(detector_results))
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
