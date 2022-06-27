from imutils import paths
import face_recognition
import pickle
import cv2
import os
import numpy as np
import time
from datetime import datetime, timedelta

def match_users_using_webcam():
    face_data_path = "./FaceDatabase/face_embeddings_2.pickle"
    casc_face_path = "./Required/haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(casc_face_path)
    face_data = pickle.loads(open(face_data_path, "rb").read())
    print("Streaming started!")
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(face_data["encodings"],
                                                     encoding, tolerance=0.5)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = face_data["names"][i]
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)


            # update the list of names
            names.append(name)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def match_user(img_rgb):
    face_data_path = "./FaceDatabase/face_embeddings_lala.pickle"
    face_data = pickle.loads(open(face_data_path, "rb").read())
    # img = cv2.imread(img_path)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    embedding = get_embedding(img_rgb)
    if embedding == -1:
        print("No face / Multiple face found!")
        return
    matches = False
    for k, v in face_data.items():
        dist = np.linalg.norm(v-embedding[0])
        #matches = face_recognition.compare_faces(v, embedding)
        # found match if euclidean distance < 0.6
        if dist < 0.5:
            return k
        # if matches:
        #     return k
    return "No match found !"

def get_embedding(img_rgb):

    boxes = face_recognition.face_locations(img_rgb, model='hog')
    if len(boxes) > 1 or len(boxes) == 0:
        return -1
    encoding = face_recognition.face_encodings(img_rgb, boxes)
    return encoding

def add_user(img_rgb, name):
    # check for existing database
    database_path = "./FaceDatabase/face_embeddings.pickle"
    if os.path.exists(database_path):
        with open(database_path, 'rb') as db:
            face_data = pickle.load(db)
            if name in face_data:
                print(f"User {name} already exists!")
                return
    else:
        # create new database
        face_data = {}
    # compute the facial embedding for the face
    encoding = get_embedding(img_rgb)
    if encoding == -1:
        print("Multiple face found / No face found!")
        return

    #save the encoding(numpy array) into a dictionary
    face_data[name] = encoding[0]
    #use pickle to save data into a file for later use
    f = open(database_path, "wb")
    f.write(pickle.dumps(face_data))
    f.close()
    print(f"user {name} added successfully!")

def add_multiple_users(image_paths):
    face_data_path = "./FaceDatabase/face_embeddings.pickle"
    face_data_2_path = "./FaceDatabase/face_embeddings_2.pickle"

    face_data = {}
    known_names = []
    known_encodings = []
    # loop over the image paths
    for(i, image_path) in enumerate(image_paths):
        #extract name from path
        name = image_path.split(os.path.sep)[-2]
        #print(name)
        #load images & convert from bgr to rgb
        image = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = get_embedding(img_rgb)
        if encoding == -1:
            print("Multiple face found or No face found!!")
            break
        #print(encoding)
        #print(encoding[0].shape)

        known_encodings.append(encoding[0])
        known_names.append(name)

        #save the encoding(numpy array) into a dictionary
        face_data[name] = encoding[0]
        print(f"User {name} added!")

    # save data in another format
    face_data_2 = {"encodings":known_encodings, "names": known_names}

    #use pickle to save data into a file for later use
    f = open(face_data_path, "wb")
    f.write(pickle.dumps(face_data))
    f.close()
    # save data in another format
    f = open(face_data_2_path, "wb")
    f.write(pickle.dumps(face_data_2))
    f.close()

if __name__ == "__main__":

    # # get paths of each file in folder named Images & add multiple users
    # image_paths = list(paths.list_images('./Images'))
    # add_multiple_users(image_paths)

    # # add new user
    # name = "rukon"
    # img_path = "./Images/rukon/rukon.jpg"
    # img = cv2.imread(img_path)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add_user(img_rgb, name)

    # match user from photo
    image_paths = list(paths.list_images('./TestImages'))
    for image_path in image_paths:

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        match = match_user(img_rgb)
        print(image_path.split('/')[-1])
        print(match)

    # # match user from webcam
    # match_users_using_webcam()