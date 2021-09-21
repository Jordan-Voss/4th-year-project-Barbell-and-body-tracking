from ..application.user import User


def test_new_user_with_fixture(new_user):
    assert new_user.email == "Jordan.voss2@mail.dcu.ie"
    assert new_user.password != "Pa$$word1"
    assert new_user.f_name == "Jordan"
    assert new_user.l_name == "Voss"
    assert new_user.username == "Vossj2"


def test_new_user():
    user = User("Jordan", "Voss", "Jordan.voss2@mail.dcu.ie", "Pa$$word1", "Vossj2")
    assert user.email == "Jordan.voss2@mail.dcu.ie"
    assert user.password != "Pa$$word1"
    assert user.f_name == "Jordan"
    assert user.l_name == "Voss"
    assert user.username == "Vossj2"
