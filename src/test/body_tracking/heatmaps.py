import cv2
# import time
import numpy as np
# import math
from application.body_functions.show_heatmap import show_heatmap as heatmap
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import glob, os

filecount = 0
protoFile = "../../../../application/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "../../../../application/pose/mpi/pose_iter_160000.caffemodel"
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
POSE_SCORES = {
    "HEAD": 0,
    "NECK": 0,
    "RSHOULDER": 0,
    "RELBOW": 0,
    "RHAND": 0,
    "LSHOULDER": 0,
    "LELBOW": 0,
    "LHAND": 0,
    "RHIP": 0,
    "RKNEE": 0,
    "RANKLE": 0,
    "LHIP": 0,
    "LKNEE": 0,
    "LANKLE": 0,
    "CHEST": 0
}
os.chdir("test_images/squat/")
for pose in POSE_NAMES:
    filecount = 0
    for file in glob.glob("*.jpg"):
        filecount += 1
        print(filecount, pose)
        # print(file)
        img = cv2.imread(file)
        # img = cv2.imread("../deadlift_bad.jpeg")
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

        # Generate Heatmap to ensure we are getting all of the points
        i = POSE_NAMES[pose]
        probability_map = output[0, i, :, :]
        _, probability, _, point = cv2.minMaxLoc(probability_map)
        x = (imgWidth * point[0]) / W
        y = (imgHeight * point[1]) / H
        print(x, y)
        heatmap(probability_map, img)
        my_input = input(f"Is this {pose} point correct (y/n): ")
        if my_input == "y":
            POSE_SCORES[pose] += 1
    with open("../squat.txt", "a") as filea:
        outstring = str(pose) +" " + str((POSE_SCORES[pose] / filecount) * 100) + "\n"
        filea.write(outstring)
