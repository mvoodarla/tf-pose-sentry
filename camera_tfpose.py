import os
import cv2
from base_camera import BaseCamera
from pose_model import PoseModel

class Camera(BaseCamera):
    video_source = 1
    model = PoseModel()

    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():

        while True:
            # read current frame
            img, num_humans = Camera.model.get_prediction()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
