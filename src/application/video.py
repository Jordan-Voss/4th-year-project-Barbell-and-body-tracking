# File to Test OpenCv without having to go through entire journey hosted on localhost
from collections import deque
import cv2
# from flask.globals import current_app
import imutils
import numpy as np
import math


def more_x_movement(history):
    x_displacement = 0
    y_displacement = 0
    i = 0
    while i < len(history):
        x_displacement += history[i][0]
        y_displacement += history[i][1]
        i += 1
    if x_displacement - y_displacement >= 5:
        return True
    else:
        return


def is_inflection_point(velocity, changed):
    if changed:
        if velocity < 0:
            return True
        else:
            return False
    else:
        if velocity >= 0:
            return True
        else:
            return False


def check_for_rep(history, is_concentric, last_direc):
    p = 0
    first = False
    second = False
    concentric = False
    displacement = 0
    first_displacement = 0
    second_displacement = 0
    velocities = []
    error = 0
    if len(history) < 10:
        return False, (0, 0, 0)
    for position in range(1, 8):
        displacement += history[-position][2]
    if displacement > 0:
        concentric = True
    elif displacement < 0:
        concentric = False
    else:
        return (False, (0, 0, 0))
    while True:
        p += 1
        if p > len(history):
            break
        if not first:
            if is_inflection_point(history[-p][2], concentric) and first_displacement > 200:

                first = True
            else:
                if is_inflection_point(history[-p][2], concentric):
                    if error > 3:
                        break
                    error += 1
                    continue
                else:
                    first_displacement += abs(history[-p][1])
                    if concentric:
                        velocities.append(abs(history[-p][2]))
                    continue

        if not second:
            # Count at least 1 second phase point before first inflection and 200mm of displacement
            # or we"re on the last point in history
            if (is_inflection_point(history[-p][2], not concentric) and second_displacement > 200) or (p == len(history) and second_displacement > 200):
                second = True
            else:
                second_displacement += abs(history[-p][1])
                if not concentric:
                    velocities.append(abs(history[-p][2]))
                continue

        # All this criteria should give us a high probability of counting a rep
        # Move more than 100mm, difference between eccentric and concentric displacement < 200mm
        # if first and second and abs(second_displacement - first_displacement) < 100:
        #     if concentric:
        #         displacement = first_displacement
        #     else:
        #         displacement = second_displacement

        avg_vel = sum(velocities) / len(velocities)
        peak_vel = max(velocities)
        print(is_concentric, last_direc)
        if is_concentric is False and last_direc is True:
            return(True, (avg_vel, peak_vel, concentric))
    return(False, (0.0, 0.0, 0))


bar_speed = 0
rest_time = 0
is_moving = False
rep_checked = False
concentric = False
last_direction = concentric
# x-move = 0
processed = False
# current_app.logger.info("Processing Video")
fps = 20
fin = False
upper_colour_range = (255, 250, 250)
lower_colour_range = (204, 0, 0)
past_points = deque(maxlen=32)
# displacement = 0
# velocity_ms = 0
y_velocity_ms = 0
avg_velocity = 0
count = 0
rep_count = 0
lastx = None
lasty = None
radius = None
peak_velocity = 0
bb_radius = 25
direction = ""
avg_velocities = []
velocities = []
history = []
peak_vel = []
rep = False
# fourcc = cv2.VideoWriter_fourcc(*"AVC1")
cap = cv2.VideoCapture('../static/video/dully.mov')
# out = cv2.VideoWriter("processess_video.mp4", 0x7634706d, fps, (640, 480))
if cap.isOpened() is False:
    print("Nah")

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        # set the frame size to we can determine the pixels per metre
        # frame = imutils.resize(frame, width=800, height=600)
        # remove noise and convert to HSV color space
        noise_remove = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(noise_remove, cv2.COLOR_BGR2HSV)
        # create a mask for the selected colour to track it
        mask = cv2.inRange(hsv, upper_colour_range, lower_colour_range)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find the outline of the circular end of the bar and a variable for the centre
        edges = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edges = imutils.grab_contours(edges)
        centre = None
        # if an outline is found
        if len(edges) > 0:
            # find the largest outline from the masked hsv image (which will be the end of the bar) and the minimun circle that encloses it
            outline = max(edges, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(outline)
            if radius is None:
                radius = r
                mmppx = bb_radius / radius
                print(f"first radius {radius}... mm per pixel {mmppx}")
            if lastx is None:
                lastx = x
                lasty = y
            # get coordinates of centre of circle
            m = cv2.moments(outline)
            # centre = (x, y)
            centre = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
            # proceed if radius meets size requirements
            pixel_dist_x = lastx - x
            pixel_dist_y = lasty - y
            lastx = x
            lasty = y
            # mm_dist_x = pixel_dist_x * mmppx
            mm_dist_y = pixel_dist_y * mmppx
            y_velocity_ms = mm_dist_y * fps / 1000
            if r > 10:
                # sys.stdout.flush
                # draw the circle and centre
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, centre, 5, (0, 0, 255), -1)
                # cv2.rectangle(frame, centre, 5, (0, 128, 255), -1)
                past_points.appendleft(centre)
                if mm_dist_y > radius / 4:
                    is_moving = True
                    rep_checked = False
                    distance_moved = math.sqrt((pixel_dist_x ** 2) + (pixel_dist_y ** 2)) * mmppx
                    velocity_ms = distance_moved * fps / 1000
                    rest_time = 0
                    if y_velocity_ms < 0.01 and y_velocity_ms > -0.01:
                        is_moving = False
                        y_velocity_ms = 0
                        rest_time += 1

                history.append((centre[0], centre[1], y_velocity_ms))

                rep, info = check_for_rep(history, concentric, last_direction)
                fin = more_x_movement(history)

                if rep is True:
                    rep_count += 1
                    history = []
                    avg_velocity = info[0]
                    avg_velocities.append(info[1])
                    peak_vel.append(info[0])
                    if rep_count == 1:
                        avg_velocity = info[0]
                        peak_velocity = info[1]
                    else:
                        avg_velocity = avg_velocities[-1]
                        peak_velocity = peak_vel[-1]
                        v_loss = (avg_velocities[0] - avg_velocities[-1]) / avg_velocities[0] * 1000
                        if v_loss > 20:
                            fin = True
            for i in np.arange(1, len(past_points)):
                if past_points[i - 1] is None or past_points[i] is None:
                    continue
                # check if enough frames have been captured to see a direction change
                if count >= 10 and i == 1 and past_points[-10] is not None:
                    # reinitialise the direction variable for when the bar is stationary
                    direction = ""
                    last_direction = concentric
                    # check for significant movement up or down
                    if past_points[-10][1] > past_points[i][1]:
                        direction = "Concentric"
                        concentric = True
                    else:
                        direction = "Eccentric"
                        concentric = False
                    # use more points when checking for a rep as a small movement could qualify as a rep
                    if len(past_points) > 20:
                        if past_points[-20][1] > past_points[i][1]:
                            concentric = True
                        else:
                            concentric = False
                cv2.line(frame, past_points[i - 1], past_points[i], (0, 0, 255), 5)
        cv2.putText(frame, f"Peak Velocity {peak_velocity}m/s", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
        cv2.putText(frame, f"Rep: {rep_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
        cv2.putText(frame, f"Average Velocity {avg_velocity}m/s", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
        cv2.putText(frame, f"EndSet {fin}", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
        cv2.imshow('FRAME', frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    count += 1
bar_speed = avg_velocities[-1]
print(bar_speed)
cap.release()
