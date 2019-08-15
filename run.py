import cv2
from pose_model import PoseModel

mod = PoseModel()

while True:
    cv2.imshow('im', mod.get_prediction())
    cv2.waitKey(1)
