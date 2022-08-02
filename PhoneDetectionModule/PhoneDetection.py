from statistics import mode
import torch
import numpy as np
import torch
import cv2

YOLOV5_WEIGHTS_PATH = r"F:\ExamProctoringSystem\PhoneDetectionModule\yolov5\runs\train\exp2\weights\best.pt"
YOLOV5_PATH = r"F:\ExamProctoringSystem\PhoneDetectionModule\yolov5"


def detect_phone_using_webcam():
    model = torch.hub.load(YOLOV5_PATH, 'custom', path=YOLOV5_WEIGHTS_PATH,
                           source='local', force_reload=True)  # local repo

    # OPEN CAMERA
    cap = cv2.VideoCapture(0)
    result = None
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model(frame)
        #cv2.imshow('frame', frame)
        frame = result.render()[0]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def detect_phone_from_img(img_rgb, model):

    result = model(img_rgb)
    # result type: <class 'models.common.Detections'>
    # check yolov5>models>common file for details
    # print(result.classnames())
    preds = result.pred
    print("##################### ",  preds, "\n")
    print(preds[0].tolist()) # [] or [[x, y, w, h, conf, label]]
    names = result.names
    # print(preds[0])
    s = ""
    for pred in preds:
        if pred.shape[0]:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                s += f"{names[int(c)]}"  # add to string
    # returns "Mobile_phone" if detected !
    return s

def detect_phone_from_img_upg(img_rgb, model):

    result = model(img_rgb)
    # result type: <class 'models.common.Detections'>
    # check yolov5>models>common file for details
    # print(result.classnames())
    preds = result.pred
    # print("##################### ",  preds, "\n")
    ans = preds[0].tolist()
    # print(ans) # [] or [[x, y, w, h, conf, label]]
    if len(ans) == 0 or ans[0][4] < .6 :
        return 0, 0, 0, 0, 0, 0
    else:
        ans = ans[0]
        phone_x = int(ans[0])
        phone_y = int(ans[1])
        phone_width = int(ans[2])
        phone_height = int(ans[3])
        phone_conf = ans[4]
        phone = 1 if phone_conf > 0.6 else 0
        return phone, phone_x, phone_y, phone_width, phone_height, phone_conf



if __name__ == "__main__":
    #detect_phone_using_webcam()

    # detect phone from rgb img
    model = torch.hub.load(YOLOV5_PATH, 'custom', path=YOLOV5_WEIGHTS_PATH,
                           source='local', force_reload=True)
    cap = cv2.VideoCapture(0)
    while True:

        _, frame = cap.read()
        frame = cv2.flip(frame, flipCode=2)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        phone, ph_x, ph_y, ph_w, ph_h, ph_conf = detect_phone_from_img_upg(img_rgb, model)
        print(phone, ph_x, ph_y, ph_w, ph_h, ph_conf)
        cv2.rectangle(frame, (ph_x, ph_y), (ph_x + ph_w , ph_y + ph_h), (0, 255, 0), 2)
        # cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
