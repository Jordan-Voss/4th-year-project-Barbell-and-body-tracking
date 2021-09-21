from ..wsgi import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):

    __tablename__ = "account"

    uid = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(255))
    l_name = db.Column(db.String(255))
    email = db.Column(db.String(255))
    password = db.Column(db.String(255))
    username = db.Column(db.String(255))
    # created_on = db.Column(db.String(255))

    def __init__(self, f_name, l_name, email, password, username):
        self.f_name = f_name
        self.l_name = l_name
        self.email = email
        self.password = generate_password_hash(password)
        self.username = username
        # self.created_on = created_on

    def set_password(self, password):
        self.password = generate_password_hash(
            password,
            method="sha256"
        )

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def get_id(self):
        return self.uid

    def is_authenticated(self):
        return self.authenticated

    def __repr__(self):
        return f"<User {self.username}>"
