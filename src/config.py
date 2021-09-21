from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, ".env"))


class Config:
    """Set Flask configuration from environment variables."""

    FLASK_APP = "wsgi.py"
    FLASK_ENV = environ.get("FLASK_ENV")
    SECRET_KEY = environ.get("SECRET_KEY")

    # Flask-Assets
    LESS_BIN = environ.get("LESS_BIN")
    ASSETS_DEBUG = environ.get("ASSETS_DEBUG")
    LESS_RUN_IN_DEBUG = environ.get("LESS_RUN_IN_DEBUG")

    # Static Assets
    STATIC_FOLDER = "static"
    TEMPLATES_FOLDER = "templates"
    COMPRESSOR_DEBUG = environ.get("COMPRESSOR_DEBUG")

    # Flask-SQLAlchemy
    SQLALCHEMY_DATABASE_URI = "postgresql://prnjoiwsycubhh:67d9374bcb6114a4a2dae1fc7a25f2f20b63f0efdd76190d4b5d2fa1ad745027@ec2-54-217-236-206.eu-west-1.compute.amazonaws.com:5432/d7jfdetlbm26sc"
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
# class Config(object):
#     DEBUG = True
#     SQLALCHEMY_TRACK_MODIFICATIONS = True
#     SECRET_KEY = os.urandom(24)
#     SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
