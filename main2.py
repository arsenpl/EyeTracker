import cv2
import numpy as np
import mediapipe as mp
from math import hypot, dist

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
font = cv2.FONT_HERSHEY_PLAIN
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmarks_points:

        left_points = []
        right_points = []
        landmarks = landmarks_points[0].landmark
        right_indx=[398,384,385,386,387,388,466,163,249,390,373,374,380,381]
        left_indx=[33,246,161,160,159,158,157,173,154,153,145,144,163,7]
        for id, landmark in enumerate(landmarks[i] for i in left_indx):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            #cv2.circle(frame, (x,y), 3, (0,255,0))
            left_points.append([x,y])
        for id, landmark in enumerate(landmarks[i] for i in right_indx):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            #cv2.circle(frame, (x,y), 3, (0,255,0))
            right_points.append([x,y])
        eye_points=[left_points[0],left_points[4],left_points[7],left_points[10],
                    right_points[0],right_points[3],right_points[8],right_points[11]]
        #cv2.line(frame, (eye_points[0][0],eye_points[0][1]),(eye_points[2][0],eye_points[2][1]), (0,0,255), 1)
        #cv2.line(frame, (eye_points[1][0], eye_points[1][1]), (eye_points[3][0], eye_points[3][1]), (0, 0, 255), 1)
        #cv2.line(frame, (eye_points[4][0], eye_points[4][1]), (eye_points[6][0], eye_points[6][1]), (0, 0, 255), 1)
        #cv2.line(frame, (eye_points[5][0], eye_points[5][1]), (eye_points[7][0], eye_points[7][1]), (0, 0, 255), 1)

        ver_len_left= int(dist(eye_points[1],eye_points[3]))
        hor_len_left = int(dist(eye_points[0], eye_points[2]))
        hor_len_right = int(dist(eye_points[4], eye_points[6]))
        ver_len_right = int(dist(eye_points[5], eye_points[7]))
        if ver_len_left!=0 and hor_len_left!=0:
            ratio_left=hor_len_left/ver_len_left
        if ver_len_right!=0 and hor_len_right!=0:
            ratio_right=hor_len_right/ver_len_right
        ratio=(ratio_left+ratio_right)/2
        if ratio_left>7 and ratio_right>7:
            cv2.putText(frame, "Both BLINK!", (50,150), font, 2, (0,0,255))
        elif ratio_left>11 and ratio_right<11:
            cv2.putText(frame, "Left BLINK!", (50,150), font, 2, (0,0,255))
        elif ratio_left<11 and ratio_right>11:
            cv2.putText(frame, "Right BLINK!", (50,150), font, 2, (0,0,255))

        left_points=np.array(left_points)
        right_points = np.array(right_points)
        print(left_points)

        minx=np.min(left_points[:,0])
        maxx=np.max(left_points[:,0])
        miny = np.min(left_points[:,1])
        maxy = np.max(left_points[:,1])




        mask = np.zeros((frame_h, frame_w), np.uint8)
        cv2.polylines(mask, [left_points], True, 255, 2)
        cv2.fillPoly(mask, [left_points],255)
        left_eye = cv2.bitwise_and(gray,gray,mask=mask)

        gray_eye = left_eye[miny:maxy, minx:maxx]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        gray_eye = cv2.resize(gray_eye, None, fx=5, fy=5)
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)

        cv2.imshow("Gray eye", gray_eye)
        cv2.imshow("Threshold", threshold_eye)
        cv2.imshow("Left eye", left_eye)

    cv2.imshow("Face tracking", frame)
    key = cv2.waitKey(10)

    # cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
cam.release()