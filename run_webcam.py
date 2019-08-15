import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

if __name__ == '__main__':

    w, h = model_wh('432x368')
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(216, 184))
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    ret_val, image = cam.read()
    count = 0
    while True:
        ret_val, image = cam.read()
        image = cv2.flip(image, 1)
        image_h, image_w = image.shape[:2]
        
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        print(len(humans))
        
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.imshow('tf-pose-estimation result', image)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
