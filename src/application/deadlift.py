from ..wsgi import db


class DeadLift(db.Model):

    __tablename__ = 'deadlift'

    versionid = db.Column(db.Integer, primary_key=True)
    userid = db.Column(db.Integer, unique=True)
    version0 = db.Column(db.String(255))
    version1 = db.Column(db.String(255))
    version2 = db.Column(db.String(255))

    def __init__(self, userid, version0, version1, version2):
        self.userid = userid
        self.version0 = version0
        self.version1 = version1
        self.version2 = version2
