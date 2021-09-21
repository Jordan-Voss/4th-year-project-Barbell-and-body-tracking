import cv2
# import time
import numpy as np
# import math
# from application.body_functions.show_heatmap import show_heatmap as heatmap
from ....application.body_functions.functions import getAngleC, getDistance, getMidPoint, plotPoint, print_pose_elements
from ....application.body_functions.deadlift.check_points import get_point_estimations
import matplotlib
matplotlib.use("TKAgg")


def deadlift_body_track(videourl, white, protoFile, weightsFile):
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
    score = 0
    back_angles = []
    bar_not_over_mid_foot_count = 0
    knee_hips_rate_is_not_equal = False
    head_length = None
    bar_distance_threshold = 55
    back_angle_threshold = 10
    knee_angles = []
    hip_angles = []
    knee_hips_angle_threshold = 20
    forearm_left = None
    forearm_right = None
    foot_size = 0
    scores = {
        "Bar Position": False,
        "Back Angle": False,
        "Knee and Hip Extension": False
    }

    # Read the video with opencv
    cap = cv2.VideoCapture(videourl)
    # fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWhite = cv2.imread(white)
    frameWidth1 = frameWhite.shape[1]
    frameHeight1 = frameWhite.shape[0]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('static/video/body.mp4', fourcc, fps, (frame_width, frame_height))
    out2 = cv2.VideoWriter('static/video/white.mp4', fourcc, fps, (frameWidth1, frameHeight1))
    while True:
        ret, frame = cap.read()
        if ret is True:
            # framecopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            probability_threshold = 0.1
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

            # Generate Heatmap to ensure we are getting all of the points
            # i = 10
            # probability_map = output[0, i, :, :]
            # heatmap(probability_map, frame)

            # Place to store the points detected
            pts = []
            for i in range(15):
                probability_map = output[0, i, :, :]
                minimum, probability, minlocation, point = cv2.minMaxLoc(probability_map)
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
            print_pose_elements(Pn)

            if None not in (pts[POSE_NAMES["RANKLE"]], pts[POSE_NAMES["RHIP"]], pts[POSE_NAMES["RHAND"]], pts[POSE_NAMES["RELBOW"]], pts[POSE_NAMES["LHAND"]], pts[POSE_NAMES["LELBOW"]], pts[POSE_NAMES["LSHOULDER"]], pts[POSE_NAMES["RSHOULDER"]], pts[POSE_NAMES["NECK"]]):
                # ============================================================================================
                # SCORE CHECK 1
                # Check that the par is positioned over mid foot
                # ============================================================================================
                # Get average forearm length to estimate foot size
                forearm_right = getDistance(pts[POSE_NAMES["RHAND"]], pts[POSE_NAMES["RELBOW"]])
                forearm_left = getDistance(pts[POSE_NAMES["LHAND"]], pts[POSE_NAMES["LELBOW"]])
                foot_size = (forearm_right + forearm_left) / 2
                fs2 = head_length * .85
                print('FOOTSIZE1:', foot_size, "FOOTSIZE2:", fs2)
                # Get the midpoint of the hands incase the video is at a slight angle and add it to pts
                hand_point = getMidPoint(pts, POSE_NAMES, "RHAND", "LHAND")
                pts.append(hand_point)
                # plotPoint(frame, hand_point, 17)
                # plotPoint(frameWhite, hand_point, 17)

                ankle_point = getMidPoint(pts, POSE_NAMES, "LANKLE", "RANKLE")
                pts.append(ankle_point)
                # plotPoint(frame, ankle_point, 18)
                # plotPoint(frameWhite, ankle_point, 18)

                bar_x_distance_ankles = abs(hand_point[0] - ankle_point[0])
                print("BAR X DISTANCE FROM ANKLES: ", bar_x_distance_ankles)

                if (fs2 / 2) + bar_distance_threshold >= bar_x_distance_ankles >= (fs2 / 2) - bar_distance_threshold:
                    print("Bar in correct Position", bar_x_distance_ankles)
                else:
                    bar_not_over_mid_foot_count += 1
                    print("Bar not in correct Position", bar_x_distance_ankles)
                # print(bar_x_distance_ankles, score, ankle_point[0], foot_size / 2)

                # ============================================================================================
                # SCORE CHECK 2
                # Check that the athlete's back is straight
                # ============================================================================================
                hips_point = getMidPoint(pts, POSE_NAMES, "LHIP", "RHIP")
                pts.append(hips_point)
                # plotPoint(frame, hips_point, 19)
                # plotPoint(frameWhite, hips_point, 19)

                shoulders_point = getMidPoint(pts, POSE_NAMES, "LSHOULDER", "RSHOULDER")
                pts.append(shoulders_point)
                # plotPoint(frame, shoulders_point, 20)
                # plotPoint(frameWhite, shoulders_point, 20)

                # plotPoint(frame, pts[POSE_NAMES["CHEST"]], 21)
                # plotPoint(frame, pts[POSE_NAMES["NECK"]], 50)

                # hl2 = getDistance(pts[POSE_NAMES["HEAD"]], shoulders_point)
                # hl2 = (hl2 + (getDistance(pts[POSE_NAMES["HEAD"]], pts[POSE_NAMES["NECK"]])))/2
                # print("HEADLENGTH1: ", head_length, "HEADLENGTH2: ", hl2)

                back_angle_chest = getAngleC(hips_point, pts[POSE_NAMES["CHEST"]], pts[POSE_NAMES["NECK"]])
                if shoulders_point == pts[POSE_NAMES["NECK"]]:
                    shoulders_point = shoulders_point[0] + 1, shoulders_point[1] + 1
                back_angle_shoulder = getAngleC(hips_point, shoulders_point, pts[POSE_NAMES["NECK"]])
                back_angles.extend([back_angle_chest, back_angle_shoulder])

                # if max(back_angle_chest, back_angle_shoulder) <= back_angle_threshold:
                #     print("Back angle correct", max(back_angle_chest, back_angle_shoulder))
                # else:
                #     back_angle_not_within_threshold = True
                #     print("Back angle not Correct", max(back_angle_chest, back_angle_shoulder))

                # cv2.putText(frame, f"Chest Angle:  {back_angle_chest}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
                # cv2.putText(frame, f"Shoulder Angle:  {back_angle_shoulder}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
                cv2.putText(frame, f"Score:  {score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
                cv2.putText(frameWhite, f"Chest Angle:  {back_angle_chest}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
                cv2.putText(frameWhite, f"Shoulder Angle:  {back_angle_shoulder}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
                cv2.putText(frameWhite, f"Score:  {score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
                # cv2.imshow('copy', frame)
                # cv2.waitKey()

                # ============================================================================================
                # SCORE CHECK 3
                # Check that the athlete's knees and hips extend at the same time
                # ============================================================================================
                knees_point = getMidPoint(pts, POSE_NAMES, "LKNEE", "RKNEE")
                print(hips_point, ankle_point, knees_point, pts[POSE_NAMES["NECK"]])
                # knee_angle = 1
                # plotPoint(frame, knees_point, 999)
                # plotPoint(frameWhite, knees_point, 999)
                if ankle_point == knees_point:
                    knees_point = knees_point[0] + 1, knees_point[1] + 1
                knee_angle = getAngleC(hips_point, ankle_point, knees_point)
                hip_angle = getAngleC(knees_point, pts[POSE_NAMES["NECK"]], hips_point)
                knee_angles.append(knee_angle)
                hip_angles.append(hip_angle)

                print("KNEE ANGLE: ", knee_angle, "HIP ANGLE: ", hip_angle)
            else:
                print("Not Enough points")
                # return {'Bar Position': False, 'Depth': False, 'Knee and Hip Extension': False}
            print(pts)
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if pts[partA] and pts[partB]:
                    # cv2.line(frame, pts[partA], pts[partB], (0, 255, 255), 2)

                    cv2.line(frame, pts[partA], pts[partB], (0, 255, 255), 2)
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
    k = 1
    while k < len(knee_angles):
        diff_knees = abs(knee_angles[k] - knee_angles[k - 1])
        diff_hips = abs(hip_angles[k] - hip_angles[k - 1])
        # print(knee_angles[k], hip_angles[k])
        if knee_hips_angle_threshold >= abs(diff_knees - diff_hips):
            print('yes', k)
        else:
            print("NO", k)
            knee_hips_rate_is_not_equal = True
            print(diff_knees, diff_hips)
        k += 1
    if bar_not_over_mid_foot_count <= 7:
        score += 1
        scores["Bar Position"] = True
    average_back_angle = sum(back_angles) / len(back_angles)
    if average_back_angle <= back_angle_threshold:
        score += 1
        scores["Back Angle"] = True
    if not knee_hips_rate_is_not_equal:
        score += 1
        scores["Knee and Hip Extension"] = True
    # print("SCORE: ", score)
    # print("AVG BACK ANGLE: ", average_back_angle)
    # print("BAR POSITION: ", bar_not_over_mid_foot_count)
    # print(knee_angles, hip_angles)
    print(scores)
    return scores
