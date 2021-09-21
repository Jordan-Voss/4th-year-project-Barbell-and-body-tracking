import pytest
from ...application.body_functions.functions import getAngleC, angleC, getDistance, getMidPoint
pts = [(135, 73), (187, 104), (-135, -3), (-17, -4), (-12235, 45543), (-1007, -7464), (2, 1), (0, 1)]
POSE_NAMES = {
        "POSITIVE1": 0,
        "POSITIVE2": 1,
        "NEGATIVE1": 2,
        "NEGATIVE2": 3,
        "LARGE1": 4,
        "LARGE2": 5,
        "SMALL1": 6,
        "SMALL2": 7
        }


def test_getMidPoint_all_positive():
    point = getMidPoint(pts, POSE_NAMES, "POSITIVE1", "POSITIVE2")
    assert (161, 88) == point


def test_getMidPoint_not_enough_inputs():
    with pytest.raises(TypeError):
        getMidPoint(pts, POSE_NAMES)


def test_getMidPoint_too_many_inputs():
    with pytest.raises(TypeError):
        getMidPoint(pts, POSE_NAMES, "POSITIVE1", "POSITIVE2", "NEGATIVE1")


def test_getMidPoint_incorrect_type():
    with pytest.raises(TypeError):
        getMidPoint(POSE_NAMES, "POSITIVE1", pts, "NEGATIVE2")


def test_getMidPoint_all_negative():
    point = getMidPoint(pts, POSE_NAMES, "NEGATIVE1", "NEGATIVE2")
    assert (-76, -3) == point


def test_getMidPoint_large_ints():
    point = getMidPoint(pts, POSE_NAMES, "LARGE1", "LARGE2")
    assert (-6621, 19039) == point


def test_getMidPoint_small_ints():
    point = getMidPoint(pts, POSE_NAMES, "SMALL1", "SMALL2")
    assert (1, 1) == point


def test_getAngleC_all_positive():
    A = (4, 4)
    B = (6, 5)
    C = (2, 9)
    assert 85.236 == round(getAngleC(A, B, C), 3)


def test_getAngleC_all_negative():
    A = (-5, -10)
    B = (-2, -7)
    C = (-2, -3)
    assert 23.199 == round(getAngleC(A, B, C), 3)


def test_getAngleC_not_enough_inputs():
    with pytest.raises(TypeError):
        getAngleC((4, 4), (2, 9))


def test_getAngleC_too_many_inputs():
    with pytest.raises(TypeError):
        getAngleC((4, 4), (6, 5), (2, 9), (8, 65))


def test_getAngleC_incorrect_type():
    with pytest.raises(TypeError):
        getAngleC(("4", 4), (6, 5), (2, 9))


def test_angleC_all_positive():
    a = 1
    b = 2
    c = 3
    assert 180 == angleC(a, b, c)


def test_angleC_large_ints():
    a = 3808
    b = 2878
    c = 1000
    assert 6.365 == round(angleC(a, b, c), 3)


def test_angleC_negative_ints():
    a = -3808
    b = -2878
    c = -1000

    assert 6.365 == round(angleC(a, b, c), 3)


def test_angleC_not_enough_inputs():
    with pytest.raises(TypeError):
        angleC(-3808, -2878)


def test_angleC_too_many_inputs():
    with pytest.raises(TypeError):
        angleC(-3808, -2878, -1000, 161)


def test_angleC_incorrect_type():
    with pytest.raises(TypeError):
        angleC("-3808", -2878, -1000, 161)


def test_getDistance_all_positive():
    pointA = [4, 3]
    pointB = [3, 2]
    assert 1.414 == round(getDistance(pointA, pointB), 3)


def test_getDistance_all_negative():
    pointA = [-6, -1]
    pointB = [-3, -2]
    assert 3.162 == round(getDistance(pointA, pointB), 3)


def test_getDistance_positive_negative():
    pointA = [1, 6]
    pointB = [-3, -2]
    assert 8.944 == round(getDistance(pointA, pointB), 3)


def test_getDistance_large_ints():
    pointA = [1000000, 3000000]
    pointB = [2000000, 4000000]
    assert 1414213.562 == round(getDistance(pointA, pointB), 3)


def test_getDistance_small_ints():
    pointA = [.034, .3]
    pointB = [.001, .004587]
    assert 0.297 == round(getDistance(pointA, pointB), 3)


def test_getDistance_large_small_ints():
    pointA = [.034, 34509827465265]
    pointB = [.001, 82837475673272]
    assert 48327648208007 == round(getDistance(pointA, pointB), 3)


def test_getDistance_not_enough_inputs():
    with pytest.raises(TypeError):
        getDistance([.001, 82837475673272])


def test_getDistance_too_many_inputs():
    with pytest.raises(TypeError):
        getDistance([.034, 34509827465265], [.001, 82837475673272], [.034, 34509827465265])


def test_getDistance_incorrect_type():
    with pytest.raises(TypeError):
        getDistance([".034", 34509827465265], [.001, 82837475673272])
