from ..application.squat import Squat
from ..application.bench import Bench
from ..application.deadlift import DeadLift
from flask import Blueprint, render_template, url_for, flash, redirect
# , current_app
from flask_login import login_user, login_required, logout_user
from ..wsgi import login_manager, db
from sqlalchemy import exc
from werkzeug.security import check_password_hash
from .forms import LoginForm, RegisterForm
from .user import User
from datetime import datetime

# Blueprint Configuration
auth_bp = Blueprint(
    "auth_bp", __name__,
    template_folder="templates",
    static_folder="static"
)


def get_time():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        try:
            new_user = User(f_name=form.f_name.data, l_name=form.l_name.data, email=form.email.data, password=form.password.data, username=form.username.data)
            db.session.add(new_user)
            db.session.commit()
            id = new_user.get_id()
            default = "2.5, 2.25, 2, 1.75, 1.5, 1.3, 1, 0.75, 0.5, 0.3"
            squat = Squat(userid=id, version0=default, version1=default, version2=default)
            bench = Bench(userid=id, version0=default, version1=default, version2=default)
            deadlift = DeadLift(userid=id, version0=default, version1=default, version2=default)
            db.session.add(squat)
            db.session.add(bench)
            db.session.add(deadlift)
            db.session.commit()
            flash("Created User {} {} with username {}".format(form.f_name.data, form.l_name.data, form.username.data))
            # current_app.logger.info(f"Created User {form.f_name.data} {form.l_name.data} with username {form.username.data}")
            return redirect("/login")
        except exc.IntegrityError:
            flash("User {} already exists. Please Log In".format(form.username.data))
            # current_app.logger.warning(f"User {form.username.data} already Exists")
            return redirect("/login")
    return render_template("register.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        flash("Login requested for user {}, remember me= {}".format(form.username.data, form.remember.data))
        if user:
            if check_password_hash(user.password, form.password.data):
                logout_user()
                login_user(user, remember=form.remember.data)
                flash("User {} Logged in successfully !".format(form.username.data))
                # current_app.logger.info(f"User {form.username.data} Logged in successfully !")
                return redirect("/")
            flash("Invalid Username or Password")
            # current_app.logger.warning("Failure Logging In..... Redirecting")
        return redirect("/")
    return render_template("login.html", form=form)


@auth_bp.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return redirect("/login")


@login_manager.user_loader
def load_user(user_id):
    """Check if user is logged-in on every page load."""
    return User.query.get(user_id)


@login_manager.unauthorized_handler
def unauthorized():
    """Redirect unauthorized users to Login page."""
    flash("You must be logged in to view that page.")
    return redirect(url_for("auth_bp.login"))
