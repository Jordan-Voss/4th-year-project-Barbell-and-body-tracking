from application.awsS3 import list_files
from application.body_functions.deadlift.deadlift import deadlift_body_track
from application.body_functions.squat.squat import squat_body_track
from application.body_functions.bench.bench import bench_body_track
lift = input()
if lift == 's' or lift == "S":
    squat_body_track("body_functions/squat_bar.mov", "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt", "pose/mpi/pose_iter_160000.caffemodel")
elif lift == 'd' or lift == "D":
    deadlift_body_track("body_functions/deadlift.mov")
elif lift == 'b' or lift == "B":
    bench_body_track("body_functions/bench_bar.mov", "../static/img/White.jpg", "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt", "pose/mpi/pose_iter_160000.caffemodel")
elif lift == 'aws':
    list_files('bbtrack-bucket', 'q-video')
