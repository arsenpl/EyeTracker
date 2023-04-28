import cv2
import numpy as np
import mediapipe as mp
import pyautogui as ag
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
#eyeball = mp.solutions.
screen_w, screen_h = ag.size()
cv2.namedWindow("Eye tracker", cv2.WINDOW_NORMAL)

# Create a black image with the same dimensions as the full screen window
img = np.zeros((1080, 1920, 3), np.uint8)

# Define the size of the heatmap
heatmap_size = (500, 500)

# Create an empty numpy array to store the heatmap
heatmap = np.zeros(heatmap_size)
heat_points=[]

i=0
while True:
    cv2.imshow('Eye tracker', img)
    _, frame = cam.read()
    frame = cv2.flip(frame,1)
    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = face_mesh.process(rgb_frame)
    landmarks_points=output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmarks_points:
        points=[]

        landmarks = landmarks_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x,y), 3, (0,255,0))
            points.append([x,y])
            #print(x,y)
        cv2.line(frame, (points[0][0],points[0][1]),(points[2][0],points[2][1]), (0,0,255), 1)
        cv2.line(frame, (points[1][0], points[1][1]), (points[3][0], points[3][1]), (0,0,255), 1)

        c_x_1 = (points[0][0]+points[2][0])/2
        c_y_1 = (points[0][1]+points[2][1])/2

        c_x_2 = (points[1][0] + points[3][0]) / 2
        c_y_2 = (points[1][1] + points[3][1]) / 2

        c_x = int((c_x_1+c_x_2) / 2)
        c_y = int((c_y_1+c_y_2) / 2)

        heat_points.append((c_y,c_x))


        if id==1:
            screen_x = screen_w/frame_w * x
            screen_y = screen_h/frame_h * y
            #ag.moveTo(screen_x, screen_y)
        left=[landmarks[145], landmarks[159]]
        right=[landmarks[374],landmarks[386]]
        #print(left[0].x,left[0].y)
        #cv2.line(frame, (left[0].x,left[0].y), (left[1].x,left[1].y), (0, 255, 255), 1)
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
            if (left[0].y-left[1].y)<0.004:
                print("Left clicked")
        for landmark in right:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
            if (right[0].y - right[1].y) < 0.004:
                print("Right clicked")
                #ag.click()
                #ag.sleep(1)


    cv2.imshow("Eye cam", frame)
    key = cv2.waitKey(1)


    # If the space bar is pressed, show a red dot
    if key == ord('s'):

        # Draw a red dot in the top-left corner
        cv2.circle(img, (50, 50), 10, (0, 0, 255), -1)
        print("check 1")
        # Wait for 3 seconds
        if i>30:
            img.fill(0)
            cv2.circle(img, (1870, 50), 10, (0, 0, 255), -1)
        # Erase the dot


        # Draw a red dot in the top-right corner

    #cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


# Loop through each point and add it to the heatmap
for point in heat_points:
    x, y = point
    if x < heatmap_size[0] and y < heatmap_size[1]:
        heatmap[x, y] += 1

# Create a plot of the heatmap
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()



cam.release()