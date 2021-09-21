from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from .config import Config
from sqlalchemy import MetaData, create_engine
# import logging
# import datetime
# import os
# import sys


# now = datetime.datetime.now()
# fname = now.strftime("logs/info_%d-%m-%Y.log")
app = Flask(__name__)
# logging.basicConfig(filename=fname, level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")
# if 'DYNO' in os.environ:
#     app.logger.addHandler(logging.StreamHandler(sys.stdout))
#     app.logger.setLevel(logging.ERROR)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config.from_object(Config)
app.config["SECRET_KEY"] = "thisissupposedtobesecret"
SQLALCHEMY_DATABASE_URI = "postgresql://prnjoiwsycubhh:67d9374bcb6114a4a2dae1fc7a25f2f20b63f0efdd76190d4b5d2fa1ad745027@ec2-54-217-236-206.eu-west-1.compute.amazonaws.com:5432/d7jfdetlbm26sc"
engine = create_engine(SQLALCHEMY_DATABASE_URI)
metadata = MetaData(engine)
db = SQLAlchemy()
db.init_app(app)

with app.app_context():
    from .application import routes
    from .application import user_authenticate
    app.register_blueprint(routes.main_bp)
    app.register_blueprint(user_authenticate.auth_bp)
    # db.create_all()

if __name__ == "__main__":
    # Run app on correct port
    app.run(port=5000, debug=True, threaded=True)
