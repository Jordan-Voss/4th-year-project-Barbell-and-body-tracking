from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, ".env"))


class TestConfig:
    """Set Flask configuration from environment variables."""

    FLASK_APP = "wsgi.py"
    FLASK_ENV = environ.get("FLASK_ENV")
    SECRET_KEY = environ.get("SECRET_KEY")
    WTF_CSRF_ENABLED = False
    # Flask-Assets
    LESS_BIN = environ.get("LESS_BIN")
    ASSETS_DEBUG = environ.get("ASSETS_DEBUG")
    LESS_RUN_IN_DEBUG = environ.get("LESS_RUN_IN_DEBUG")

    # Static Assets
    STATIC_FOLDER = "static"
    TEMPLATES_FOLDER = "templates"
    COMPRESSOR_DEBUG = environ.get("COMPRESSOR_DEBUG")

    # Flask-SQLAlchemy
    SQLALCHEMY_DATABASE_URI = "postgres://tglbtqbngcsdlf:7891363c85f616cd0c6e9d3784dbe298eeec1c7c20e620d36b22d1111798134c@ec2-54-228-250-82.eu-west-1.compute.amazonaws.com:5432/d84i4kmgf9rc6b"
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
# class Config(object):
#     DEBUG = True
#     SQLALCHEMY_TRACK_MODIFICATIONS = True
#     SECRET_KEY = os.urandom(24)
#     SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
