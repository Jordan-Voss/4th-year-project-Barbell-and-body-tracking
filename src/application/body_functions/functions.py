import math
import cv2


def plotPoint(img, point, number):
    cv2.circle(img, (point[0], point[1]), 3, (0, 255, 255), thickness=1, lineType=cv2.FILLED)
    cv2.putText(img, f"{number}", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)


def getMidPoint(pts, POSE_NAMES, partA, partB):
    x_coord = (pts[POSE_NAMES[partA]][0] + pts[POSE_NAMES[partB]][0])/2
    y_coord = (pts[POSE_NAMES[partA]][1] + pts[POSE_NAMES[partB]][1])/2
    return (int(x_coord), int(y_coord))


def getDistance(pointA, pointB):
    # get the distance between 2 points
    return math.sqrt(((pointA[0] - pointB[0]) ** 2) + ((pointA[1] - pointB[1]) ** 2))


def angleC(a, b, c):
    if a == 0:
        a = 0.001
    if b == 0:
        b = 0.001
    if c == 0:
        c = 0.001

    return math.degrees(math.acos((c**2 - b**2 - a**2)/(-2.0 * a * b)))


def getAngleC(pointA, pointB, pointC):
    AB_dist = getDistance(pointA, pointB)
    BC_dist = getDistance(pointB, pointC)
    AC_dist = getDistance(pointA, pointC)
    return(max(angleC(BC_dist, AC_dist, AB_dist), angleC(AC_dist, AB_dist, BC_dist)))


def print_pose_elements(Pn):
    for k, v in Pn.items():
        print(k, v)


def diff(a, b):
    return abs(a - b)
