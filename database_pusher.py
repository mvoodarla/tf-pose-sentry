from google.cloud import storage
from firebase import firebase
import os
import cv2
import datetime

class DatabasePusher:
   def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="tf-pose-sentry-b7a3b2d45290.json"
        self.firebase = firebase.FirebaseApplication('https://tf-pose-sentry.firebaseio.com/')
        self.client = storage.Client()
        self.bucket = self.client.get_bucket('tf-pose-sentry.appspot.com')

   def push(self, image):
        cv2.imwrite('temp_image.png', image)
        imageBlob = self.bucket.blob("/")
        imageBlob = self.bucket.blob(str(datetime.datetime.now()).split('.')[0])
        imageBlob.upload_from_filename('temp_image.png')
