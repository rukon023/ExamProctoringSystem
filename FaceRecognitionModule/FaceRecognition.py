from tkinter.tix import Tree
from imutils import paths
#import face_recognition
import pickle
import cv2
import os
#import dlib
import sys
import time
from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector

BASE_PATH = r"E:\FinalProject\Finalcode\ExamProctoringSystem\FaceRecognitionModule"
image_paths = list(paths.list_images('Images'))
DATABASE_PATH = os.path.join(BASE_PATH, "FaceDatabase")
predictor_path = "Required/shape_predictor_5_face_landmarks.dat"
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
metrics = ["cosine", "euclidean", "euclidean_l2"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace",
          "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
# face detectors
detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
test_img_path = "E:\FinalProject\Finalcode\ExamProctoringSystem\FaceRecognitionModule\TestImages\muza.jpg"

# # face recognition
# df = DeepFace.find(test_img_path, db_path=DATABASE_PATH, model_name=models[2], detector_backend=backends[2], align=False,
#                    prog_bar=False)
# print(f"lala{df}lala")

# img = DeepFace.detectFace(test_img_path, detector_backend = detectors[4], target_size=(224, 224))
# print(type(img), img.shape)


def add_user_using_webcam():
    name = input("Enter Name: ")
    db_path = DATABASE_PATH
    user_path = os.path.join(db_path, name)
    print(user_path)
    if os.path.exists(user_path):
        print("User already exists!")
        return
    else:
        os.makedirs(user_path)

    face_cascade = cv2.CascadeClassifier(os.path.join(BASE_PATH,
                                                      'Required/haarcascade_frontalface_default.xml'))

    cap = cv2.VideoCapture(0)
    i = 3
    face_found = False

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1, 0)

        cv2.putText(frame, "Keep Your Face in front of Camera",
                    (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, 'Starting', (260, 270), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.putText(frame, str(i), (290, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (255, 255, 255), 3)

        i -= 1
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)

        if cv2.waitKey(1) == ord('q'):
            break

        if i < 0:
            break

    i = 4
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1, 0)
        img = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)

        text = f"Number of photo required: {i+1}"
        cv2.putText(frame, text,
                    (int((frame.shape[1]-len(text))/4),
                     100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        if len(face) == 1:
            for(x, y, w, h) in face:
                roi = frame[y-10:y+h+10, x-10:x+w+10]
                fh, fw = roi.shape[:2]
                # make sure that face roi is of required height & shape
                if fh < 20 and fw < 20:
                    continue

                cv2.rectangle(frame, (x-10, y-10),
                              (x+w+10, y+h+10), (255, 0, 0), 2)
                save_dir = os.path.join(user_path, f"{name}_{i}.jpg")
                cv2.imwrite(save_dir, img)

                i -= 1
                time.sleep(2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) and i < 0:
            break
        # if cv2.waitKey(1) == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

    for f in os.listdir(db_path):
        if f.endswith(".pkl"):
            os.remove(os.path.join(db_path, f))


def match_user():
    
    test_img_path = "E:\FinalProject\Finalcode\ExamProctoringSystem\FaceRecognitionModule\TestImages\Photo.jpg"
    df = DeepFace.find(test_img_path, DATABASE_PATH,
                           model_name='Facenet',
                           distance_metric="euclidean_l2",
                           )
    print(df["identity"][0])
    pass


def match_user_using_webcam():
    DeepFace.stream(DATABASE_PATH, enable_face_analysis=False)
    # pass


if __name__ == "__main__":
    choice = int(input(
        f"Choices: \n1. Add New User(1)\n2. Match User using webcam\n3. Match User(3):"))
    if choice == 1:
        add_user_using_webcam()
    elif choice == 2:
        match_user_using_webcam()
    elif choice == 3:
        match_user()
    # pass
