from re import X
import numpy as np
import cv2
import copy
import mediapipe as mp

classFile = 'coco.names'

with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)

while True:
    hasPhone = 0
    # get video frame (always BGR format!)
    ret, frame = cap.read()
    if (ret):
        # copy image to draw on
        img = copy.copy(frame) 
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        if len(classIds) != 0:
            lastConfidence = 0
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                isConfident = confidence>0.5
                if(classNames[classId-1]=="cell phone") and isConfident:
                    if hasPhone == 0:
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        lastConfidence=confidence
                    if hasPhone==1 and confidence>lastConfidence:
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        lastConfidence=confidence
                    hasPhone = 1
        results = pose.process(img)
        if results.pose_landmarks:
            if hasPhone==0:
                mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
        # show the original image with drawings in one window
        cv2.imshow('Original image', img)

        # deal with keyboard input
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    else:
        print('Could not start video camera')
        break

cap.release()
cv2.destroyAllWindows()

