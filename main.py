import cv2
import mediapipe as mp
import numpy as np
import time
import math
import imutils
# import mediapipe_video as v

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

mp_drawing2 = mp.solutions.drawing_utils
mp_drawing_styles2 = mp.solutions.drawing_styles
mp_pose2 = mp.solutions.pose
i=0
videos = ["./videos/test22.mp4","./videos/test8.mp4","./videos/test17.mp4"]
# For webcam input:
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(videos[i])

pTime = 0
left_arm_pose = [11, 13, 15]
right_arm_pose = [12, 14, 16]
left_leg_pose = [11, 23, 25]
right_leg_pose = [12, 24, 26]
left_knee_pose = [23, 25, 27]
right_knee_pose = [24, 26, 28]
right_shoulder_pose = [14, 12, 24]
left_shoulder_pose = [13, 11, 23]



drawing_spec = mp_drawing.DrawingSpec(color = (224, 224, 224), thickness = 3, circle_radius = 2)
drawing_spec1 = mp_drawing.DrawingSpec(color = (102, 255, 0), thickness = 3, circle_radius = 2)
drawing_spec2 = mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 3, circle_radius = 2)
def get_left_arm_pose(lmList1, lmlist2):

    x1, y1 = lmList1[11][1:]
    x2, y2 = lmList1[13][1:]
    x3, y3 = lmList1[15][1:]

    angle1 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle1 = int(angle1)

    x1, y1 = lmlist2[11][1:]
    x2, y2 = lmlist2[13][1:]
    x3, y3 = lmlist2[15][1:]

    angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle2 = int(angle2)

    a = abs(angle1 - angle2)
    print("===right arm angle===")
    print(angle1, angle2)
    print()
    if a >= 20:
        return (11, 13), (13, 15)
    else:
        return True


def get_right_arm_pose(lmList1, lmlist2):

    x1, y1 = lmList1[12][1:]
    x2, y2 = lmList1[14][1:]
    x3, y3 = lmList1[16][1:]

    angle1 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    angle1 = int(angle1)

    x1, y1 = lmlist2[12][1:]
    x2, y2 = lmlist2[14][1:]
    x3, y3 = lmlist2[16][1:]

    angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle2 = int(angle2)


    a = abs(angle1 - angle2)
    print("===left arm angle===")
    print(angle1, angle2)
    print()
    if a >= 20:
        return (12, 14), (14, 16)
    else:
        return True

def get_left_leg_pose(lmList1, lmlist2):

    x1, y1 = lmList1[11][1:]
    x2, y2 = lmList1[23][1:]
    x3, y3 = lmList1[25][1:]

    angle1 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    angle1 = int(angle1)

    x1, y1 = lmlist2[11][1:]
    x2, y2 = lmlist2[23][1:]
    x3, y3 = lmlist2[25][1:]

    angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle2 = int(angle2)

    a = abs(angle1 - angle2)
    print("===right leg angle===")
    print(angle1, angle2)
    print()

    if a <= 20:
        return True
    else:
        return (11, 23), (23, 25)

def get_right_leg_pose(lmList1, lmlist2):

    x1, y1 = lmList1[12][1:]
    x2, y2 = lmList1[24][1:]
    x3, y3 = lmList1[26][1:]

    angle1 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    angle1 = int(angle1)

    x1, y1 = lmlist2[12][1:]
    x2, y2 = lmlist2[24][1:]
    x3, y3 = lmlist2[26][1:]

    angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle2 = int(angle2)

    a = abs(angle1 - angle2)
    print("===left leg angle===")
    print(angle1, angle2)
    print()
    if a >= 20:
        return (12, 24), (24, 26)
    else:
        return True

def get_left_knee_pose(lmList1, lmlist2):

    x1, y1 = lmList1[23][1:]
    x2, y2 = lmList1[25][1:]
    x3, y3 = lmList1[27][1:]

    angle1 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    angle1 = int(angle1)

    x1, y1 = lmlist2[23][1:]
    x2, y2 = lmlist2[25][1:]
    x3, y3 = lmlist2[27][1:]

    angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle2 = int(angle2)

    a = abs(angle1 - angle2)
    print("===right knee angle===")
    print(angle1, angle2)
    print()
    if a >= 20:
        return (23, 25), (25, 27)
    else:
        return True

def get_right_knee_pose(lmList1, lmlist2):

    x1, y1 = lmList1[24][1:]
    x2, y2 = lmList1[26][1:]
    x3, y3 = lmList1[28][1:]

    angle1 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    angle1 = int(angle1)

    x1, y1 = lmlist2[24][1:]
    x2, y2 = lmlist2[26][1:]
    x3, y3 = lmlist2[28][1:]

    angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle2 = int(angle2)

    a = abs(angle1 - angle2)
    print("===left knee angle===")
    print(angle1, angle2)
    print()
    if a >= 20:
        return (24, 26), (26, 28)
    else:
        return True


def get_right_shoulder_pose(lmList1, lmlist2):

    x1, y1 = lmList1[14][1:]
    x2, y2 = lmList1[12][1:]
    x3, y3 = lmList1[24][1:]

    angle1 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle1 = int(angle1)

    x1, y1 = lmlist2[14][1:]
    x2, y2 = lmlist2[12][1:]
    x3, y3 = lmlist2[24][1:]

    angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle2 = int(angle2)

    a = abs(angle1 - angle2)
    print("===right shoulder angle===")
    print(angle1, angle2)
    print()
    if a >= 20:
        return (12, 14), (12, 24)
    else:
        return True

def get_left_shoulder_pose(lmList1, lmlist2):

    x1, y1 = lmList1[13][1:]
    x2, y2 = lmList1[11][1:]
    x3, y3 = lmList1[23][1:]

    angle1 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle1 = int(angle1)

    x1, y1 = lmlist2[13][1:]
    x2, y2 = lmlist2[11][1:]
    x3, y3 = lmlist2[23][1:]

    angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle2 = int(angle2)

    a = abs(angle1 - angle2)
    print("===left shoulder angle===")
    print(angle1, angle2)
    print()
    if a >= 20:
        return (11, 13), (11, 23)
    else:
        return True

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    with mp_pose2.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose2:

      while cap.isOpened():
        success, image = cap.read()
        success2, image2 = cap2.read()
        image = cv2.flip(image, 1)
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        if not success2:
          print("Ignoring empty camera frame.")
          cap2.release()
          i += 1
          if i == 3:
              break
          cap2 = cv2.VideoCapture(videos[i])
          success2, image2 = cap2.read()
          #cap.release()
          #cap2.release()
          #cv2.destroyAllWindows()
          # If loading a video, use 'break' instead of 'continue'.
          #break

        image.flags.writeable = False
        image2.flags.writeable = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        results = pose.process(image)
        results2 = pose2.process(image2)

        image.flags.writeable = True
        image2.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

        lmList_cam = [[0] * 3 for i in range(33)]
        lmList_video = [[0] * 3 for i in range(33)]
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = lm.x*w, lm.y*h
                lmList_cam[id] = [id, cx, cy]

        # print(lmList_cam)
        if results2.pose_landmarks:
            for id, lm in enumerate(results2.pose_landmarks.landmark):
                h, w, c = image2.shape
                cx, cy = lm.x*w, lm.y*h
                lmList_video[id] = [id, cx, cy]

        # print(lmList_video)
        # print(lmList_cam, lmList_video)
        #print(get_left_arm_pose(lmList_cam, lmList_video))
        #print(get_right_arm_pose(lmList_cam, lmList_video))
        #print(get_left_leg_pose(lmList_cam, lmList_video))

        sum = 0
        precision = 0

        a = get_left_arm_pose(lmList_cam, lmList_video)
        b = get_right_arm_pose(lmList_cam, lmList_video)
        c = get_left_leg_pose(lmList_cam, lmList_video)
        d = get_right_leg_pose(lmList_cam, lmList_video)
        e = get_left_knee_pose(lmList_cam, lmList_video)
        f = get_right_knee_pose(lmList_cam, lmList_video)
        g = get_right_shoulder_pose(lmList_cam, lmList_video)
        h = get_left_shoulder_pose(lmList_cam, lmList_video)

        sum += 8

        pose_connections_list = []
        pose_true_connections_list = []

        if a != True:
            cv2.putText(image,"left arm",(20,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
            pose_connections_list.append(a[0])
            pose_connections_list.append(a[1])
            precision += 1
        elif a:
            pose_true_connections_list.append((11,13))
            pose_true_connections_list.append((13,15))
        if b != True:
            cv2.putText(image, "right arm", (20, 90),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
            pose_connections_list.append(b[0])
            pose_connections_list.append(b[1])
            precision += 1
        else:
            pose_true_connections_list.append((12,14))
            pose_true_connections_list.append((14,16))
        if c != True:
            cv2.putText(image, "left leg", (20, 130),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
            pose_connections_list.append(c[0])
            pose_connections_list.append(c[1])
            precision += 1
        else:
            pose_true_connections_list.append((11,23))
            pose_true_connections_list.append((23,25))
        if d != True:
            cv2.putText(image, "right leg", (20, 170),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
            pose_connections_list.append(d[0])
            pose_connections_list.append(d[1])
            precision += 1
        else:
            pose_true_connections_list.append((12,24))
            pose_true_connections_list.append((24,26))
        if e != True:
            cv2.putText(image, "left knee", (20, 210),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
            pose_connections_list.append(e[0])
            pose_connections_list.append(e[1])
            precision += 1
        else:
            pose_true_connections_list.append((23,25))
            pose_true_connections_list.append((25,27))
        if f != True:
            cv2.putText(image, "right knee", (20, 250),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
            pose_connections_list.append(f[0])
            pose_connections_list.append(f[1])
            precision += 1
        else:
            pose_true_connections_list.append((24,26))
            pose_true_connections_list.append((26,28))
        if g != True:
            cv2.putText(image, "left shoulder", (20, 290),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
            pose_connections_list.append(g[0])
            pose_connections_list.append(g[1])
            precision += 1
        else:
            pose_true_connections_list.append((14,12))
            pose_true_connections_list.append((12,24))
        if h != True:
            cv2.putText(image, "right shoulder", (20, 330),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
            pose_connections_list.append(h[0])
            pose_connections_list.append(h[1])
            precision += 1
        else:
            pose_true_connections_list.append((13,11))
            pose_true_connections_list.append((11,23))

        print("pose_connections_list".upper())
        print(pose_connections_list)



        mp_drawing.draw_landmarks(image, results.pose_landmarks, pose_true_connections_list,
                                  connection_drawing_spec=drawing_spec1)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, pose_connections_list,
                                  connection_drawing_spec=drawing_spec2)
        mp_drawing2.draw_landmarks(image2, results2.pose_landmarks, mp_pose2.POSE_CONNECTIONS, connection_drawing_spec=drawing_spec)


        #image = imutils.resize(image, width=1000)
        #image2 = imutils.resize(image2, width=500)
        image = cv2.resize(image, dsize=(950, 1000), interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, dsize = (950,1000),interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, str((sum-precision)/sum * 100)+'%', (700, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow('A3_skeleton_cam', image)
        cv2.imshow('A3_skeleton_video', image2)

        cv2.waitKey(1)

