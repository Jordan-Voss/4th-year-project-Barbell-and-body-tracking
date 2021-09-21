from re import T
import cv2
# import time
import numpy as np
import imutils
# import math
# from application.body_functions.show_heatmap import show_heatmap as heatmap
from ....application.body_functions.functions import getDistance, getMidPoint, plotPoint, print_pose_elements, diff
from ....application.body_functions.deadlift.check_points import get_point_estimations
import matplotlib
matplotlib.use("TKAgg")


def bench_body_track(videourl, white, protoFile, weightsFile):
    # define the body parts and the pairs that link together
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
    # Define some variables
    nPoints = 15
    greenUpper = (75, 100, 80)
    greenLower = (75, 100, 50)
    score = 0
    bar_past_chest = False
    head_length = None
    forearm_left = None
    forearm_right = None
    foot_size = 0
    foot = None
    feet_point = None
    bench_point = None
    hips_score = True
    feet_score = True
    feet_score_count = 3
    hip_score_count = 5
    scores = {
        "Bar Past Chest": False,
        "Hips on Bench": False,
        "Feet on Floor": False
    }


    # Read the video with opencv
    cap = cv2.VideoCapture(videourl)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWhite = cv2.imread(white)
    frameWidth1 = frameWhite.shape[1]
    frameHeight1 = frameWhite.shape[0]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter("static/video/body.mp4", fourcc, fps, (frame_width, frame_height))
    out2 = cv2.VideoWriter("static/video/white.mp4", fourcc, fps, (frameWidth1, frameHeight1))
    while True:
        ret, frame = cap.read()
        if ret is True:
            # framecopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            probability_threshold = 0.05
            frameWhite = cv2.imread(white)

            # read the network
            network = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

            w = 368
            h = 368
            inputBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (w, h), (0, 0, 0), swapRB=False, crop=False)
            network.setInput(inputBlob)
            output = network.forward()
            H = output.shape[2]
            W = output.shape[3]
            print(output.shape)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, greenLower, greenUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), _) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Generate Heatmap to ensure we are getting all of the points
            # i = 10
            # probability_map = output[0, i, :, :]
            # heatmap(probability_map, frame)

            # Place to store the points detected
            pts = []
            for i in range(15):
                probability_map = output[0, i, :, :]
                _, probability, _, point = cv2.minMaxLoc(probability_map)
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H

                if probability > probability_threshold:
                    pts.append((int(x), int(y)))
                else:
                    pts.append(None)
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
            if head_length is None:
                try:
                    head_length = getDistance(pts[POSE_NAMES["HEAD"]], pts[POSE_NAMES["NECK"]])
                except Exception:
                    head_length = None
                if head_length:
                    foot_size = head_length * .85
                else:
                    try:
                        forearm_right = getDistance(pts[POSE_NAMES["RHAND"]], pts[POSE_NAMES["RELBOW"]])
                    except Exception:
                        print("couldnt get one or both forearm lengths.")
                    try:
                        forearm_left = getDistance(pts[POSE_NAMES["LHAND"]], pts[POSE_NAMES["LELBOW"]])
                    except Exception:
                        print("couldnt get one or both forearm lengths.")
                    if forearm_left and forearm_right:
                        foot_size = (forearm_right + forearm_left) / 2
                    elif forearm_left and not forearm_right:
                        foot_size = forearm_left
                    elif forearm_right and not forearm_left:
                        foot_size = forearm_right
                    head_length = (foot_size / 85) * 100
            # head_length = getDistance(pts[POSE_NAMES["HEAD"]], pts[POSE_NAMES["NECK"]])
            # plotPoint(frame, pts[POSE_NAMES["HEAD"]], 17)
            # plotPoint(frame, pts[POSE_NAMES["NECK"]], 17)
            # Get the Bar Distance from the ankle
            pts = get_point_estimations(pts, POSE_NAMES, head_length)

            # if None not in (pts[POSE_NAMES["RANKLE"]], pts[POSE_NAMES["RHIP"]], pts[POSE_NAMES["RELBOW"]]):
            # ============================================================================================
            # SCORE CHECK 1
            # Check that the athlete's hips stay on the bench
            # ============================================================================================
            hips_point = getMidPoint(pts, POSE_NAMES, "LHIP", "RHIP")
            if not bench_point:
                bench_point = hips_point
            pts.append(hips_point)
            plotPoint(frame, hips_point, 19)
            plotPoint(frameWhite, hips_point, 19)
            if diff(hips_point[1], bench_point[1]) > 150:
                if hip_score_count == 0:
                    hips_score = False
                hip_score_count -= 1
            # ============================================================================================
            # SCORE CHECK 2
            # Check that the athlete's feet don't leave the floor.
            # ============================================================================================
            feet_point = getMidPoint(pts, POSE_NAMES, "LANKLE", "RANKLE")
            if not foot:
                foot = feet_point
            pts.append(feet_point)
            plotPoint(frame, feet_point, 19)
            plotPoint(frameWhite, feet_point, 19)
            if diff(feet_point[1], foot[1]) > 30:
                if feet_score_count == 0:
                    feet_score = False
                feet_score_count -= 1

            # ============================================================================================
            # SCORE CHECK 3
            # Check that the bar doesnt go past the mid chest
            # ============================================================================================
            if pts[POSE_NAMES["CHEST"]] is not None and center is not None:
                if center[0] > pts[POSE_NAMES["CHEST"]]:
                    bar_past_chest = True

            # ============================================================================================
            # Draw the points on the white background
            # ============================================================================================

            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if pts[partA] and pts[partB]:
                    # cv2.line(frame, pts[partA], pts[partB], (0, 255, 255), 2)
                    cv2.line(frameWhite, pts[partA], pts[partB], (0, 255, 255), 2)

            out.write(frame)
            out2.write(frameWhite)
        else:
            break
        out.write(frame)
        out2.write(frameWhite)
    out.release()
    out2.release()
    cap.release()
    cv2.destroyAllWindows()
    if not bar_past_chest:
        score += 1
        scores["Bar Past Chest"] = True
    if hips_score:
        score += 1
        scores["Hips on Bench"] = True
    if feet_score:
        score += 1
        scores["Feet on Floor"] = True
    print(scores)
    return(scores)
