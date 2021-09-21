from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, FileField, SelectField
from wtforms.validators import InputRequired, Length, EqualTo


lifts = [("Squat", "Squat"), ("Bench", "Bench Press"), ("Deadlift", "Deadlift")]
colours = [("redGradient", "Red"), ("greenGradient", "Green"), ("blueGradient", "Blue"), ("yellowGradient", "Yellow")]


class LoginForm(FlaskForm):
    username = StringField("Username: ", validators=[InputRequired()], id="username-input")
    password = PasswordField("Password: ", validators=[InputRequired()], id="password-input")
    remember = BooleanField("Remember Me")


class RegisterForm(FlaskForm):
    f_name = StringField('First Name: ', validators=[InputRequired()], id='fname_input')
    l_name = StringField('Last Name: ', validators=[InputRequired()], id='lname_input')
    username = StringField('Username: ', validators=[InputRequired()], id='uname_input')
    email = StringField('Email: ', validators=[InputRequired()], id='email_input')
    password = StringField('Password', validators=[InputRequired(), Length(min=6), EqualTo('password2', message='Passwords Must Match')], id='pw_input')
    password2 = StringField('Retype Password', validators=[InputRequired(), Length(min=6)], id='pw2_input')


class VideoForm(FlaskForm):
    video = FileField()
    submit = SubmitField('Save Video')


class LiftType(FlaskForm):
    lift = SelectField(u"Lift: ", choices=lifts, id="liftselector")


class ColourSelect(FlaskForm):
    colour = SelectField(u"Colour", choices=colours, id="colours")


class RepForm(FlaskForm):
    f_rep = StringField('First Rep : ', validators=[InputRequired()], id='f_input')
    s_rep = StringField('Second Rep: ', validators=[InputRequired()], id='s_input')
    exc = StringField('Excercise: ', validators=[InputRequired()], id='exc_input')


class RPEForm(FlaskForm):
    rpe = StringField('RPE: ', validators=[InputRequired()], id='rpe_input')
