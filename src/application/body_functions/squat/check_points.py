def get_point_estimations(pts, POSE_NAMES, head_length):
    shin_length = head_length * 2
    if None not in pts:
        return pts
    try:
        # check ankle points
        if pts[POSE_NAMES["RANKLE"]] is None and pts[POSE_NAMES["LANKLE"]] is not None:
            pts[POSE_NAMES["RANKLE"]] = pts[POSE_NAMES["LANKLE"]]
        elif pts[POSE_NAMES["LANKLE"]] is None and pts[POSE_NAMES["RANKLE"]] is not None:
            pts[POSE_NAMES["LANKLE"]] = pts[POSE_NAMES["RANKLE"]]

        # check knee points
        if pts[POSE_NAMES["RKNEE"]] is None and pts[POSE_NAMES["LKNEE"]] is not None:
            pts[POSE_NAMES["RKNEE"]] = pts[POSE_NAMES["LKNEE"]]
        elif pts[POSE_NAMES["LKNEE"]] is None and pts[POSE_NAMES["RKNEE"]] is not None:
            pts[POSE_NAMES["LKNEE"]] = pts[POSE_NAMES["RKNEE"]]
        # if we have both knees and no ankles
        if pts[POSE_NAMES["LANKLE"]] is None and pts[POSE_NAMES["RANKLE"]] is None and pts[POSE_NAMES["LKNEE"]] is not None and pts[POSE_NAMES["RKNEE"]] is not None:
            pts[POSE_NAMES["LANKLE"]] = (pts[POSE_NAMES["LKNEE"]][0], pts[POSE_NAMES["LKNEE"]][1] - shin_length)
            pts[POSE_NAMES["RANKLE"]] = (pts[POSE_NAMES["RKNEE"]][0], pts[POSE_NAMES["RKNEE"]][1] - shin_length)
        # and vice versa
        elif pts[POSE_NAMES["LANKLE"]] is not None and pts[POSE_NAMES["RANKLE"]] is not None and pts[POSE_NAMES["LKNEE"]] is None and pts[POSE_NAMES["RKNEE"]] is None:
            pts[POSE_NAMES["LKNEE"]] = (pts[POSE_NAMES["LANKLE"]][0], pts[POSE_NAMES["LANKLE"]][1] + shin_length)
            pts[POSE_NAMES["RKNEE"]] = (pts[POSE_NAMES["RANKLE"]][0], pts[POSE_NAMES["RANKLE"]][1] + shin_length)
    except Exception:
        return "NOT ENOUGH LEG POINTS"
    return pts
