import cv2
# import time
import numpy as np
# import math
from application.body_functions.show_heatmap import show_heatmap as heatmap
from application.body_functions.functions import getAngleC, getDistance, getMidPoint, plotPoint, print_pose_elements
from application.body_functions.deadlift.check_points import get_point_estimations
import matplotlib
import boto3
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
# input = input("Enter IMAGE:")
img = cv2.imread(f"1.png")
# img = cv2.imread("../deadlift_bad.jpeg")
bar_distance_threshold = 55
back_angle_threshold = 10
imgcopy = np.copy(img)
imgWidth = img.shape[1]
imgHeight = img.shape[0]
probability_threshold = 0.1

# read the network
network = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

w = 368
h = 368
inputBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (w, h), (0, 0, 0), swapRB=False, crop=False)
network.setInput(inputBlob)
output = network.forward()
H = output.shape[2]
W = output.shape[3]
print(output.shape)

# Generate Heatmap to ensure we are getting all of the points
i = 10
probability_map = output[0, i, :, :]
# heatmap(probability_map, img)

# Place to store the points detected
pts = []
for i in range(15):
    probability_map = output[0, i, :, :]
    minimum, probability, minlocation, point = cv2.minMaxLoc(probability_map)
    x = (imgWidth * point[0]) / W
    y = (imgHeight * point[1]) / H

    if probability > probability_threshold:
        pts.append((int(x), int(y)))
        # print(POSE_NAMES[i], (int(x), int(y)))
    else:
        pts.append(None)
pts[6] = None
pts[7] = None
pts[4] = None
Pn = {
    "HEAD": pts[0],
    "NECK": pts[1],
    "RSHOULDER": pts[2],
    "RELBOW": pts[3],
    "RHAND": pts[4],
    "LSHOULDER": pts[5],
    "LELBOW": pts[6],
    "LHAND": pts[7],
    "RHIP": pts[8],
    "RKNEE": pts[9],
    "RANKLE": pts[10],
    "LHIP": pts[11],
    "LKNEE": pts[12],
    "LANKLE": pts[13],
    "CHEST": pts[14]
    }
print_pose_elements(Pn)
# print(getAngles(pts[1], pts[-1], pts[-4]))
head_length = getDistance(pts[POSE_NAMES["HEAD"]], pts[POSE_NAMES["NECK"]])
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]
    if pts[partA] and pts[partB]:
        cv2.line(img, pts[partA], pts[partB], (0, 255, 0), 3)
cv2.imshow('copy', img)
cv2.waitKey()
