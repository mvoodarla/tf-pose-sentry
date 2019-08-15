import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

class PoseModel:
    
    def __init__(self):
        self.w, self.h = model_wh('432x368')
        self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(216, 184))
        self.cam = cv2.VideoCapture(1)
        #self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        #self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    def get_prediction(self):
        ret_val, image = self.cam.read()
        #image = cv2.flip(image, 1)
        image_h, image_w = image.shape[:2]
        
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=4.0)
        
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        return image, len(humans)
