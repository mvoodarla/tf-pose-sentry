import os
import cv2
from base_camera import BaseCamera
from pose_model import PoseModel
from database_pusher import DatabasePusher as db
import time
import datetime

class Camera(BaseCamera):
    video_source = 1
    model = PoseModel()
    db = db()

    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        last_num_humans = 0
        last_time_pushed = 10
        while True:
            # read current frame
            img, num_humans = Camera.model.get_prediction()
            frame_info = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + " Humans: " + str(num_humans)
            cv2.putText(img, frame_info, (10, img.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            
            if (last_num_humans != num_humans or time.time() - last_time_pushed > 5) and num_humans != 0:
                Camera.db.push(img)
                last_time_pushed = time.time()

            last_num_humans = num_humans
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
