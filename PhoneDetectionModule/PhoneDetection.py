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


if __name__ == "__main__":
    detect_phone_using_webcam()

    # # detect phone from rgb img
    # model = torch.hub.load(YOLOV5_PATH, 'custom', path=YOLOV5_WEIGHTS_PATH,
    #                        source='local', force_reload=True)
    # cap = cv2.VideoCapture(0)
    # while True:

    #     _, frame = cap.read()
    #     frame = cv2.flip(frame, flipCode=2)
    #     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     class_name = detect_phone_from_img(img_rgb, model)
    #     print(class_name if class_name != "" else None)
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
