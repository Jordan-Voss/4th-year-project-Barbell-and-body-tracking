def get_point_estimations(pts, POSE_NAMES, head_length):
    shin_length = int(head_length * 2)
    # i = 0
    # while i <= 1:
    if None not in pts:
        return pts
    try:
        # check ankle points
        if pts[POSE_NAMES["RANKLE"]] is None and pts[POSE_NAMES["LANKLE"]] is not None:
            pts[POSE_NAMES["RANKLE"]] = pts[POSE_NAMES["LANKLE"]][0], pts[POSE_NAMES["LANKLE"]][1] + 1
        elif pts[POSE_NAMES["LANKLE"]] is None and pts[POSE_NAMES["RANKLE"]] is not None:
            pts[POSE_NAMES["LANKLE"]] = pts[POSE_NAMES["RANKLE"]][0], pts[POSE_NAMES["RANKLE"]][1] + 1

        # check knee points
        if pts[POSE_NAMES["RKNEE"]] is None and pts[POSE_NAMES["LKNEE"]] is not None:
            pts[POSE_NAMES["RKNEE"]] = pts[POSE_NAMES["LKNEE"]][0], pts[POSE_NAMES["LKNEE"]][1] + 1
        elif pts[POSE_NAMES["LKNEE"]] is None and pts[POSE_NAMES["RKNEE"]] is not None:
            pts[POSE_NAMES["LKNEE"]] = pts[POSE_NAMES["RKNEE"]][0], pts[POSE_NAMES["RKNEE"]][1] + 1
        # if we have both knees and no ankles
        if pts[POSE_NAMES["LANKLE"]] is None and pts[POSE_NAMES["RANKLE"]] is None and pts[POSE_NAMES["LKNEE"]] is not None and pts[POSE_NAMES["RKNEE"]] is not None:
            pts[POSE_NAMES["LANKLE"]] = (pts[POSE_NAMES["LKNEE"]][0], pts[POSE_NAMES["LKNEE"]][1] - shin_length)
            pts[POSE_NAMES["RANKLE"]] = (pts[POSE_NAMES["RKNEE"]][0], pts[POSE_NAMES["RKNEE"]][1] - shin_length)
        # and vice versa
        elif pts[POSE_NAMES["LANKLE"]] is not None and pts[POSE_NAMES["RANKLE"]] is not None and pts[POSE_NAMES["LKNEE"]] is None and pts[POSE_NAMES["RKNEE"]] is None:
            pts[POSE_NAMES["LKNEE"]] = (pts[POSE_NAMES["LANKLE"]][0], pts[POSE_NAMES["LANKLE"]][1] + shin_length)
            pts[POSE_NAMES["RKNEE"]] = (pts[POSE_NAMES["RANKLE"]][0], pts[POSE_NAMES["RANKLE"]][1] + shin_length)
        # check hand points
        if pts[POSE_NAMES["RHAND"]] is None and pts[POSE_NAMES["LHAND"]] is not None:
            pts[POSE_NAMES["RHAND"]] = pts[POSE_NAMES["LHAND"]][0], pts[POSE_NAMES["LHAND"]][1] + 1
        elif pts[POSE_NAMES["LHAND"]] is None and pts[POSE_NAMES["RHAND"]] is not None:
            pts[POSE_NAMES["LHAND"]] = pts[POSE_NAMES["RHAND"]][0], pts[POSE_NAMES["RHAND"]][1] + 1

        # check elbow points
        print("HERE")
        if pts[POSE_NAMES["RELBOW"]] is None and pts[POSE_NAMES["LELBOW"]] is not None:
            pts[POSE_NAMES["RELBOW"]] = pts[POSE_NAMES["LELBOW"]][0], pts[POSE_NAMES["LELBOW"]][1] + 1
        elif pts[POSE_NAMES["LELBOW"]] is None and pts[POSE_NAMES["RELBOW"]] is not None:
            pts[POSE_NAMES["LELBOW"]] = pts[POSE_NAMES["RELBOW"]][0], pts[POSE_NAMES["RELBOW"]][1] + 1
        # if we have both elbows and no hands
        if pts[POSE_NAMES["LHAND"]] is None and pts[POSE_NAMES["RHAND"]] is None and pts[POSE_NAMES["LELBOW"]] is not None and pts[POSE_NAMES["RELBOW"]] is not None:
            pts[POSE_NAMES["LHAND"]] = (pts[POSE_NAMES["LELBOW"]][0], pts[POSE_NAMES["LELBOW"]][1] - shin_length)
            pts[POSE_NAMES["RHAND"]] = (pts[POSE_NAMES["RELBOW"]][0], pts[POSE_NAMES["RELBOW"]][1] - shin_length)
        # and vice versa
        elif pts[POSE_NAMES["LHAND"]] is not None and pts[POSE_NAMES["RHAND"]] is not None and pts[POSE_NAMES["LELBOW"]] is None and pts[POSE_NAMES["RELBOW"]] is None:
            pts[POSE_NAMES["LELBOW"]] = (pts[POSE_NAMES["LHAND"]][0], pts[POSE_NAMES["LHAND"]][1] + shin_length)
            pts[POSE_NAMES["RELBOW"]] = (pts[POSE_NAMES["RHAND"]][0], pts[POSE_NAMES["RHAND"]][1] + shin_length)
        # Do the same for hips
        if pts[POSE_NAMES["RHIP"]] is None and pts[POSE_NAMES["LHIP"]] is not None:
            pts[POSE_NAMES["RHIP"]] = pts[POSE_NAMES["LHIP"]][0], pts[POSE_NAMES["LHIP"]][1] + 1
        elif pts[POSE_NAMES["LHIP"]] is None and pts[POSE_NAMES["RHIP"]] is not None:
            pts[POSE_NAMES["LHIP"]] = pts[POSE_NAMES["RHIP"]][0], pts[POSE_NAMES["RHIP"]][1] + 1
    except Exception:
        return "NOT ENOUGH POINTS"
        # i += 1
    return pts
    # If we can't find the wrists we can get the rough place they should be given body proportions
    # if pts[POSE_NAMES["LHAND"]] is None and pts[POSE_NAMES["RHAND"]] is None:
    #     upper_arm_length = getDistance(pts[POSE_NAMES["LSHOULDER"]], pts[POSE_NAMES["LELBOW"]])
    #     pts[POSE_NAMES["LHAND"]] = (pts[POSE_NAMES["LELBOW"]][0], pts[POSE_NAMES["LELBOW"]][1] - upper_arm_length)
    #     pts[POSE_NAMES["RHAND"]] = (pts[POSE_NAMES["LELBOW"]][0], pts[POSE_NAMES["LELBOW"]][1] - upper_arm_length)
    # if pts[POSE_NAMES["RHAND"]] is None:
    #     pts[POSE_NAMES["RHAND"]] = pts[POSE_NAMES["LHAND"]]
    # elif pts[POSE_NAMES["LHAND"]] is None:
    #     pts[POSE_NAMES["LHAND"]] = pts[POSE_NAMES["RHAND"]]
    # if pts[POSE_NAMES["RKNEE"]] is None:
    #     pts[POSE_NAMES["RKNEE"]] = pts[POSE_NAMES["LKNEE"]]
    # if pts[POSE_NAMES["RANKLE"]] is None:
    #     pts[POSE_NAMES["RANKLE"]] = pts[POSE_NAMES["LANKLE"]]
    # if pts[POSE_NAMES["LANKLE"]] is None:
    #     pts[POSE_NAMES["LANKLE"]] = pts[POSE_NAMES["RANKLE"]]
    # pts[POSE_NAMES["RELBOW"]] = pts[POSE_NAMES["LELBOW"]]
