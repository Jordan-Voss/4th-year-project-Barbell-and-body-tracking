from ..application.functions import float_list_to_string, generate_chart, get_rpe, string_to_float_list
from flask import Blueprint, render_template, request, redirect, url_for
from flask.globals import session
import boto3
# , current_app
from flask_login import current_user, login_required
from ..wsgi import db
from .forms import RPEForm, VideoForm, LiftType, ColourSelect
from .squat import Squat
from .bench import Bench
from .body_info import BodyInfo
from .deadlift import DeadLift
from .body_functions.squat import squat as sq
from .body_functions.bench import bench as bp
from .body_functions.deadlift import deadlift as dl
from ..application import video_process as vp
from ..application import polyreg as pr
from ..application import awsS3 as aws
from ..application import body_constants as bc

# Configure Blueprint to break the app into smaller reuseable components
main_bp = Blueprint(
    "main_bp", __name__,
    template_folder="../templates",
    static_folder="../static"
)


@main_bp.route("/", methods=["GET", "POST"])
def home():
    # current_app.logger.info("Homepage Loaded")
    return render_template("home.html")


@main_bp.route("/picker", methods=["GET", "POST"])
@login_required
def picker():
    liftform = LiftType()
    colourform = ColourSelect()
    if liftform.validate_on_submit() and colourform.validate_on_submit():
        lift = liftform.lift.data
        colour = colourform.colour.data
        # print(lift, colour)
        return redirect(url_for(".rec_upl", lift=lift, colour=colour))
    return render_template("lift_and_colour_picker.html", form=liftform, form2=colourform)


@main_bp.route("/video", methods=["GET", "POST"])
@login_required
def video():
    form = VideoForm()
    lift = request.args.get("lift")
    colour = request.args.get("colour")
    print("VIDEO", lift, colour)
    if form.validate_on_submit():
        # current_app.logger.info("Video Saved Successfully")
        return redirect(url_for(".upload"))
        # , lift=request.args.lift, colour=request.args["colour"]
    return render_template("record.html", form=form, name=current_user.username, lift=lift, colour=colour)


@main_bp.route("/upload", methods=["GET", "POST"])
def upload():
    lift = request.args.get("lift")
    colour = request.args.get("colour")
    print(lift, colour)
    speed = 0
    # upper_colour = colours[colour][0]
    # lower_colour = colours[colour][1]
    vid = current_user.username + f"/original/{lift}.mp4"
    videourl = f"https://bbtrack-bucket.s3-eu-west-1.amazonaws.com/{vid}"
    print("LIIIIIIIFFFFFTTT")
    print(lift)
    print(colour)
    print(vid)

    if request.method == "POST":
        video = request.files.get("video")
        file = video
        # filename = secure_filename(file.filename)
        aws.S3_upload("bbtrack-bucket", current_user.username, file, vid)
        s3 = boto3.resource('s3')
        obj = s3.Object("bbtrack-bucket", "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        prototxt = obj.get()['Body'].read()
        obj = s3.Object("bbtrack-bucket", "pose/mpi/pose_iter_160000.caffemodel")
        caffemodel = obj.get()['Body'].read()
        if lift == 'Squat':
            scores = sq.squat_body_track(videourl, "src/static/img/White.jpg",
                                         prototxt, caffemodel)
            vid = current_user.username + "/body/Squat"
        elif lift == 'Deadlift':
            scores = dl.deadlift_body_track(videourl, "src/static/img/White.jpg",
                                            prototxt, caffemodel)
            vid = current_user.username + "/body/Deadlift"
        else:
            scores = bp.bench_body_track(videourl, "src/static/img/White.jpg",
                                         prototxt, caffemodel)
            vid = current_user.username + "/body/Bench"
        print('speed')
        inf = BodyInfo(userid=current_user.uid, info=str(scores), url=lift)
        db.session.add(inf)
        db.session.commit()
        infoid = inf.get_id()
        vid = vid + "-" + str(infoid) + ".mp4"
        print(infoid, inf)
        aws.S3_upload("bbtrack-bucket", current_user.username, "dl.mp4", vid)
        speed = vp.process_video(videourl, current_user.username, colour, lift, infoid)
        if speed > 0:
            rpe = get_rpe(speed)
            chart = generate_chart(speed, rpe)
            session['chart'] = chart
            # if form.is_submitted():
            #     return redirect(url_for(".set_info", lift=lift, speed=speed, rpe=rpe))
            # else:
            #     return render_template("speed.html", name=f"{current_user.f_name} {current_user.l_name}", videourl="s3://bbtrack-bucket/q-video/video.mp4", dic={}, lift=lift, speed=speed, form=form)
            return redirect(url_for(".rpe", lift=lift, speed=speed, rpe=rpe, colour=colour, infoid=infoid))
    return render_template("speed.html", name=f"{current_user.f_name} {current_user.l_name}", videourl="s3://bbtrack-bucket/q-video/video.mp4", dic={}, lift=lift, speed=speed)


@main_bp.route("/rpe")
@login_required
def rpe():
    form = RPEForm()
    print('got form')
    lift = request.args.get("lift")
    speed = float(request.args.get("speed"))
    rpe = request.args.get("rpe")
    colour = request.args.get("colour")
    infoid = request.args.get("infoid")
    print(infoid)
    return render_template("speed.html", name=f"{current_user.f_name} {current_user.l_name}", videourl="s3://bbtrack-bucket/q-video/video.mp4",
                           dic={}, rpe=rpe, lift=lift, speed=speed, form=form, colour=colour, infoid=infoid)


@main_bp.route("/set-info", methods=["GET", "POST"])
@login_required
def set_info():
    chart = session['chart']
    lift = request.args.get("lift")
    speed = request.args.get("speed")
    rpe = request.args.get("rpe")
    infoid = request.args.get("infoid")
    print("SET INFO= ", infoid)
    if lift == "Squat":
        exercise = Squat.query.filter_by(userid=current_user.uid).first()
    elif lift == "Bench":
        exercise = Bench.query.filter_by(userid=current_user.uid).first()
    else:
        exercise = DeadLift.query.filter_by(userid=current_user.uid).first()
    v1 = exercise.version1
    v2 = exercise.version2
    v2 = string_to_float_list(v2)
    v1 = string_to_float_list(v1)
    print(lift, speed, rpe)
    x, y = pr.data_rep(v1, v2, chart)
    print("XY", x, y)
    exercise.version0 = exercise.version1
    exercise.version1 = exercise.version2
    exercise.version2 = float_list_to_string(y)
    # dt.versionid += 1
    db.session.commit()

    list1 = y.tolist()
    print(list1)
    chart_dict = {'RPE(10)': {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []},
                  'RPE(9)': {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []},
                  'RPE(8)': {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []},
                  'RPE(7)': {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []},
                  'RPE(6)': {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []}}
    count = list1[0]
    fourth = (list1[3] / count) * 100
    diff = (100 - fourth) / 4
    lst12 = [100, None, None, None, None, None, None, None, None, None]

    for i in range(1, 10):
        lst12[i] = round((lst12[i-1] - diff), 1)
    print(lst12)

    def rpe_chart(list):
        list = list[1:]
        list.append(round((list[-1] - diff), 1))
        return list

    # for x in list1:
    #     y = round(float(x / count) * 100)
    #     perclist.append(y)

    # def my_function(lst):
    #     news = []
    #     i = 0
    #     j = 1
    #     while j < len(lst):
    #         difer = lst[i] - lst[j]
    #         t = lst[i] - difer
    #         news.append(t)
    #         i = i + 1
    #         j = j + 1
    #         if i == 9:
    #             q = lst[i] - difer
    #             news.append(q)
    #     return news
    rpe9 = rpe_chart(lst12)
    chart_dict['RPE(10)'] = lst12
    chart_dict['RPE(9)'] = rpe9
    rpe8 = rpe_chart(rpe9)
    chart_dict['RPE(8)'] = rpe8
    rpe7 = rpe_chart(rpe8)
    chart_dict['RPE(7)'] = rpe7
    rpe6 = rpe_chart(rpe7)
    chart_dict['RPE(6)'] = rpe6
    return render_template("speed2.html", name=f"{current_user.f_name} {current_user.l_name}",
                           videourl="s3://bbtrack-bucket/q-video/video.mp4", dic=chart_dict, lift=lift, speed=speed, rpe=rpe, infoid=infoid)


@main_bp.route("/recordorupload")
def rec_upl():
    lift = request.args.get("lift")
    colour = request.args["colour"]
    print(lift, colour)
    return render_template("upload_record.html", lift=lift, colour=colour)


@main_bp.route("/choosetrack")
def choose():
    return render_template("choosetrack.html")


@main_bp.route("/body")
def body():
    return render_template("body.html")


@main_bp.route("/upload_file")
def upl():
    lift = request.args.get("lift")
    colour = request.args.get("colour")
    print(lift)
    return render_template("upload_file.html", lift=lift, colour=colour)


@main_bp.route("/stash")
@login_required
def stash():
    usr = current_user.username
    contents = aws.list_files('bbtrack-bucket', f'{usr}/bar')
    all = {}
    # print(contents)
    for content in contents:
        # print(content)
        a = content.split("-")
        b = a[-1].split(".")
        print(b)
        all[content] = b[0]
    print(all)

    return render_template("stash.html", list=all)


@main_bp.route("/stash-track")
@login_required
def stashtrack():
    infoid = request.args.get("infoid")
    form = BodyInfo.query.filter_by(infoid=infoid).first()
    inf = eval(form.info)
    score = 0
    scores = {}
    info = {}
    keys = []
    for key in inf:
        keys.append(key)
        if inf[key] is True:
            score += 1
            scores[key] = bc.dictionary[key]['SCORE']
            info[key] = ["Great Job!"]
        else:
            scores[key] = bc.dictionary[key]['NOSCORE']
            info[key] = bc.dictionary[key]['TIPS']
    print(scores)

    return render_template("stashtrack.html",  infoid=infoid, score=scores, tips=info, keys=keys, total=score, vid1="../static/video/bar.mp4", vid2="../static/video/body.mp4")


@main_bp.route("/form-check")
@login_required
def form():
    infoid = request.args.get("infoid")
    print(infoid, "FORMCEHCL")
    form = BodyInfo.query.filter_by(infoid=infoid).first()
    inf = eval(form.info)
    score = 0
    scores = {}
    info = {}
    keys = []
    for key in inf:
        keys.append(key)
        if inf[key] is True:
            score += 1
            scores[key] = bc.dictionary[key]['SCORE']
            info[key] = ["Great Job!"]
        else:
            scores[key] = bc.dictionary[key]['NOSCORE']
            info[key] = bc.dictionary[key]['TIPS']
    print(scores)

    return render_template("bodytrack.html", infoid=inf, score=scores, tips=info, keys=keys, total=score)


@main_bp.route("/uploaded", methods=["POST", "GET"])
def uplfile():
    form = RPEForm()
    lift = request.args.get("lift")
    print("THIS IS THE LIFT: ", lift)
    colour = request.args.get("colour")
    # print(colour)
    scores = {}
    vid = current_user.username + f"/original/{lift}.mp4"
    videourl = f"https://bbtrack-bucket.s3-eu-west-1.amazonaws.com/{vid}"
    print(vid)
    if request.method == 'POST':
        f = request.files.get("file")
        aws.S3_upload("bbtrack-bucket", current_user.username, f, vid)
        s3 = boto3.resource('s3')
        obj = s3.Object("bbtrack-bucket", "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        prototxt = obj.get()['Body'].read()
        obj = s3.Object("bbtrack-bucket", "pose/mpi/pose_iter_160000.caffemodel")
        caffemodel = obj.get()['Body'].read()
        # current_app.logger.info('file uploaded successfully')
        if lift == 'Squat':
            # pass
            scores = sq.squat_body_track(videourl, "src/static/img/White.jpg",
                                         prototxt, caffemodel)
            vid = current_user.username + "/body/Squat"
        elif lift == 'Deadlift':
            # pass
            scores = dl.deadlift_body_track(videourl, "src/static/img/White.jpg",
                                            prototxt, caffemodel)
            vid = current_user.username + "/body/Deadlift"
        elif lift == 'Bench':
            scores = bp.bench_body_track(videourl, "src/static/img/White.jpg",
                                         prototxt, caffemodel)
            vid = current_user.username + "/body/Bench"
        print("insterting to database")
        print(scores)
        inf = BodyInfo(userid=current_user.uid, info=str(scores), url=lift)
        db.session.add(inf)
        db.session.commit()
        print("added")
        infoid = inf.get_id()
        vid = vid + "-" + str(infoid) + ".mp4"
        print(infoid, inf)
        aws.S3_upload("bbtrack-bucket", current_user.username, "dl.avi", vid)
        speed = vp.process_video(videourl, current_user.username, colour, lift, infoid)
        if speed > 0:
            rpe = get_rpe(speed)
            chart = generate_chart(speed, rpe)
            session['chart'] = chart
            # if form.is_submitted():
            #     return redirect(url_for(".set_info", lift=lift, speed=speed, rpe=rpe))
            # else:
            #     return render_template("speed.html", name=f"{current_user.f_name} {current_user.l_name}", videourl="s3://bbtrack-bucket/q-video/video.mp4", dic={}, lift=lift, speed=speed, form=form)
            return redirect(url_for(".rpe", lift=lift, speed=speed, rpe=rpe, colour=colour, infoid=infoid))
        # current_app.logger.info('file processed successfully')
    return render_template("speed.html", name=f"{current_user.f_name} {current_user.l_name}",
                           videourl="s3://bbtrack-bucket/q-video/video.mp4",  dic={}, lift=lift, speed=speed, form=form, infoid=infoid)


@main_bp.errorhandler(Exception)
def internal_error(error):
    print(f"Error {error} has occured")
    return render_template("generic_error.html", e=error)
