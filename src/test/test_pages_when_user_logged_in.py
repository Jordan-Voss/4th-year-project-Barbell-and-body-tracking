from ..wsgi import app as flask_app
from ..testconfig import TestConfig
import pytest
flask_app.testing = True


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


def test_login(app, client):
    res = client.post("/login", data=dict(username="q", password="qqq"), follow_redirects=True)
    assert res.status_code == 200
    assert b"Login requested for user q, remember me= False" in res.data
    assert b"User q Logged in successfully !" in res.data


def test_video(app, client):
    res = client.post("/login", data=dict(username="q", password="qqq"), follow_redirects=True)
    assert res.status_code == 200
    assert b"Login requested for user q, remember me= False" in res.data
    assert b"User q Logged in successfully !" in res.data
    res = client.get("/video", follow_redirects=True)
    assert res.status_code == 200
    assert b"START RECORDING" in res.data
    assert b"STOP RECORDING" in res.data


def test_picker(app, client):
    res = client.post("/login", data=dict(username="q", password="qqq"), follow_redirects=True)
    assert res.status_code == 200
    assert b"Login requested for user q, remember me= False" in res.data
    assert b"User q Logged in successfully !" in res.data
    res = client.get("/picker")
    assert res.status_code == 200
    assert b"<h1>Pick Lift</h1>" in res.data

def test_stash(app, client):
    res = client.post("/login", data=dict(username="q", password="qqq"), follow_redirects=True)
    assert res.status_code == 200
    assert b"Login requested for user q, remember me= False" in res.data
    assert b"User q Logged in successfully !" in res.data
    res = client.get("/stash")
    assert res.status_code == 200
    assert b' <title>Stash</title>' in res.data

def test_recordupload(app, client):
    res = client.post("/login", data=dict(username="q", password="qqq"), follow_redirects=True)
    assert res.status_code == 200
    assert b"Login requested for user q, remember me= False" in res.data
    assert b"User q Logged in successfully !" in res.data
    res = client.get("/upload_file")
    assert res.status_code == 200
    assert b"file" in res.data
    assert b"submit" in res.data

def test_choosetrack(app, client):
    res = client.post("/login", data=dict(username="q", password="qqq"), follow_redirects=True)
    assert res.status_code == 200
    assert b"Login requested for user q, remember me= False" in res.data
    assert b"User q Logged in successfully !" in res.data
    res = client.get("/choosetrack")
    assert res.status_code == 200
    assert b"upload" in res.data
    assert b"record" in res.data
