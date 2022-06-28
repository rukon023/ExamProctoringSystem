from imutils import paths
import os
from datetime import datetime
import cv2
import dlib
from deepface import DeepFace
DATABASE_DIR = "./FaceDatabase"
image_paths = list(paths.list_images('Images'))
test_img_path = "./TestImages/opu2.jpg"

#DeepFace.stream(DATABASE_DIR, model_name='Facenet512', source=1, frame_threshold=10)
print(DeepFace.verify(test_img_path, "./FaceDatabase/opu/opu.jpg"))



