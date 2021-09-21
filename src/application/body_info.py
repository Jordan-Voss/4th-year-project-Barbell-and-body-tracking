from ..wsgi import db


class BodyInfo(db.Model):

    __tablename__ = "bodyinfo"

    infoid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    info = db.Column(db.String(255))
    userid = db.Column(db.Integer)
    url = db.Column(db.String(255))

    def __init__(self, info, userid, url):
        self.info = info
        self.userid = userid
        self.url = url

    def get_id(self):
        return self.infoid
