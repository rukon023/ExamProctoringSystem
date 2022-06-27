from imutils import paths
import face_recognition
import pickle
import cv2
import os
import dlib
import sys

image_paths = list(paths.list_images('Images'))
known_encodings = []
known_names = []

predictor_path = "Required/shape_predictor_5_face_landmarks.dat"
face_recog_model_path = "Required/dlib_face_recognition_resnet_model_v1.dat"
#loading detector model to find the faces, face landmarks model to precisely localize the face, and the face recognition model
detector = dlib.get_frontal_face_detector()
shape_pred = dlib.shape_predictor(predictor_path)
face_recog = dlib.face_recognition_model_v1(face_recog_model_path)

win = dlib.image_window()

for(i, image_path) in enumerate(image_paths):

    name = image_path.split(os.path.sep)[-2]
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    dets = detector(img_rgb, 1)
    print(f"Number of faces detected: {len(dets)}")

    if len(dets) > 1:
        print("Multiple faces found! Please select photo with single face!")
    elif len(dets) == 0:
        print("No face found!!")
    else:
        for k, d in enumerate(dets):
            print(f"Detection {k}: Left:{d.left()}, Top: {d.top()}, Right: {d.right()}, Bottom: {d.bottom()}")
            # get the landmarks/parts for the face in box d
            shape = shape_pred(img_rgb, d)

            #Draw the face landmarks on the screen
            win.clear_overlay()
            win.set_image(img_rgb)
            win.add_overlay(shape)

            #Compute 128D vector that describes the face. if 2 vector hav a euclidean distance less
            #than 0.6, then they are from the same person.
            face_descriptor = face_recog.compute_face_descriptor(img_rgb, shape)
            print(face_descriptor)

            dlib.hit_enter_to_continue()






