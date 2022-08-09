import cv2
import mediapipe as mp
import numpy as np

class PoseDetect():
    def __init__(self,minDetectionCon=0.5,minTrackingCon=0.5 ):
        self.minDetectionCon = minDetectionCon;
        self.minTrackingCon = minTrackingCon;
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def findPose(self,image,draw = True):
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        self.image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #self.image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # To improve performance
        self.image.flags.writeable = False

        # Get the result
        self.results = self.face_mesh.process(image);

        # To improve performance
        self.image.flags.writeable = True

        # Convert the color space from RGB to BGR
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        lmlist = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                
                #print("Ahnaf")

                print(x," ",y);
                lmlist.append([x,y]);
                #See where the user's head tilting
                if y > 10:
                    text = "Looking Left"
                    print("Looking left")
                elif y < -10:
                    text = "Looking Right"
                    print("looking Right");
                elif x < -10:
                    print("Looking Down")
                    text = "Looking Down"
                else:
                    text = "Forward"
                    print("Looking Forward");

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                cv2.line(image, p1, p2, (255, 0, 0), 2)

                # # Add the text on the image
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Head Pose Estimation', image)
        return lmlist
           
    



def main():
    obj = PoseDetect();
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        lmlist = obj.findPose(image);
        # for x in lmlist:
        #     if x[1]>10:
        #         print("Looking Left");
        #     elif x[1]<-10:
        #         print("Looking Right");
        #     elif x[0]<-10:
        #         print("Looking Down");
        #     else:
        #         print("Looking Forward");
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows();

if __name__ == "__main__":
    main();