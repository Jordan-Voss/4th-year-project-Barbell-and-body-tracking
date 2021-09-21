import pytest
from ..wsgi import app as flask_app
from ..testconfig import TestConfig
flask_app.testing = True

# class SomeTest(MyTest):

#     def test_something(self):

#         user = User()
#         db.session.add(user)
#         db.session.commit()

#         # this works
#         assert user in db.session

#         response = self.client.get("/")

#         # this raises an AssertionError
#         assert user in db.session


@pytest.fixture
def app():
    flask_app.config.from_object(TestConfig)
    flask_app.config["SECRET_KEY"] = "secretkeyforapptesting"
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_home(app, client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"<title>Homepage</title>" in res.data
    # client.assert_context("greeting", "hello")


def test_register(app, client):
    res = client.get("/register")
    assert res.status_code == 200


def test_login(app, client):
    res = client.post("/login")
    assert res.status_code == 200
    assert b"<title>Login</title>" in res.data


def test_video_redirect(app, client):
    res = client.get("/video", follow_redirects=True)
    assert res.status_code == 200
    assert b"<title>Login</title>" in res.data


def test_video_no_redirect(app, client):
    res = client.get("/video")
    assert res.status_code == 302


def test_recordupload_redirect(app, client):
    res = client.get("/recordorupload", follow_redirects=True)
    assert res.status_code == 200
    assert b"<title>Login</title>" in res.data

def test_recordupload_no_redirect(app, client):
    res = client.get("/recordorupload")
    assert res.status_code == 200



def test_picker(app, client):
    res = client.get("/picker")
    assert res.status_code == 302


def test_stash(app, client):
    res = client.get("/stash")
    assert res.status_code == 302


def test_capture_video(app, client):
    res = client.get("/stash")
    assert res.status_code == 302


def test_capture_video(app, client):
    res = client.get("/choosetrack")
    assert res.status_code == 200


def test_capture_video_no_redirect(app, client):
    res = client.get("/choosetrack")
    assert res.status_code == 200

def test_capture_video_redirect(app, client):
   res = client.get("/choosetrack", follow_redirects=True)
   assert res.status_code == 200
   assert b"<title>Picker</title>" in res.data

def test_upload_file_no_redirect(app, client):
    res = client.get("/upload_file")
    assert res.status_code == 200

def test_capture_video_no_redirect(app, client):
   res = client.get("/choosetrack", follow_redirects=True)
   assert res.status_code == 200
   assert b"<title>Picker</title>" in res.data

def test_body(app, client):
    res = client.get("/body")
    assert res.status_code == 200

def test_uploaded(app, client):
    res = client.get("/uploaded")
    assert res.status_code == 200