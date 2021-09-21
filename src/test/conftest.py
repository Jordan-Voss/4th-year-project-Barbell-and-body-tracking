import pytest
from ..wsgi import db
from ..application.user import User


@pytest.fixture(scope="module")
def new_user():
    user = User("Jordan", "Voss", "Jordan.voss2@mail.dcu.ie", "Pa$$word1", "Vossj2")
    return user


# @pytest.fixture(scope="module")
# def test_client():
#     flask_app = create_app()

#     # Create a test client using the Flask application configured for testing
#     with flask_app.test_client() as testing_client:
#         # Establish an application context
#         with flask_app.app_context():
#             yield testing_client  # this is where the testing happens!


@pytest.fixture(scope="module")
def init_database():
    # Create the database and the database table
    db.create_all()

    # Insert user data
    user1 = User("Jordan", "Voss", "Jordan.voss2@mail.dcu.ie", "Pa$$word1", "Vossj2", "")
    db.session.add(user1)

    # Commit the changes for the users
    db.session.commit()

    yield db  # this is where the testing happens!

    db.drop_all()
