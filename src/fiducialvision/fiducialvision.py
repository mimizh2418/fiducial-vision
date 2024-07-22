import cv2

from .config import *
from .pipeline import *


def run_pipeline():
    camera_config = load_camera_config('camera_config.json')
    fiducial_config = load_fiducial_config('fiducial_config.json')

    vid = cv2.VideoCapture(camera_config.index)
    detector = FiducialDetector(fiducial_config)

    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        print(detector.detect_fiducials(frame))
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
