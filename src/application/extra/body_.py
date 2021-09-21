import sys
import cv2
import os
from sys import platform

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    try:
        if platform == "win32":
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            sys.path.append('openpose/build/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    params = dict()
    params["model_folder"] = "openpose/models/"
    params["face"] = False
    params["hand"] = False
    params["body"] = 1

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()
    cap = cv2.VideoCapture("../static/video/dully.mov")
    out = cv2.VideoWriter("processess_video.mp4", 0x7634706d, 25, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            imageToProcess = frame
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            print("Body keypoints: \n" + str(datum.poseKeypoints))
            out.write(datum.cvOutputData)
            # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap = False
        else:
            cv2.closeAllWindows

except Exception as e:
    print(e)
    sys.exit(-1)
