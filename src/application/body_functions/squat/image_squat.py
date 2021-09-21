import cv2
# import time
import numpy as np
import imutils
import boto3
# import math
from application.body_functions.show_heatmap import show_heatmap as heatmap
from application.body_functions.functions import getAngleC, getDistance, getMidPoint, plotPoint, print_pose_elements
from application.body_functions.squat.check_points import get_point_estimations
import matplotlib
import application.awsS3 as s3
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

score = 0
s3 = boto3.resource('s3')
obj = s3.Object("bbtrack-bucket", "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
protoFile = obj.get()['Body'].read()
obj = s3.Object("bbtrack-bucket", "pose/mpi/pose_iter_160000.caffemodel")
weightsFile = obj.get()['Body'].read()
nPoints = 15
POSE_NAMES = {
    "HEAD": 0,
    "NECK": 1,
    "RSHOULDER": 2,
    "RELBOW": 3,
    "RHAND": 4,
    "LSHOULDER": 5,
    "LELBOW": 6,
    "LHAND": 7,
    "RHIP": 8,
    "RKNEE": 9,
    "RANKLE": 10,
    "LHIP": 11,
    "LKNEE": 12,
    "LANKLE": 13,
    "CHEST": 14
    }

POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
              [1, 14], [-1, 8], [8, 9], [9, 10], [-1, 11], [11, 12], [12, 13], [14, -1]]
input = input("Enter IMAGE:")
img = cv2.imread(f"{input}.png")

# read the network
network = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

w = 368
h = 368
inputBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (w, h), (0, 0, 0), swapRB=False, crop=False)
network.setInput(inputBlob)
output = network.forward()
H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
points = []
for i in range(len()):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    # Scale the point to fit on the original image
    x = (W * point[0]) / W
    y = (H * point[1]) / H
    cv2.circle(img, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
    cv2.putText(img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
# Add the point to the list if the probability is greater than the threshold
    points.append((int(x), int(y)))
    for pair in POSE_PAIRS:
        partA = pair[0]
    partB = pair[1]
    if points[partA] and points[partB]:
        cv2.line(img, points[partA], points[partB], (0, 255, 0), 3)
cv2.imshow("Output-Keypoints",img)
cv2.waitKey(0)
cv2.destroyAllWindows()