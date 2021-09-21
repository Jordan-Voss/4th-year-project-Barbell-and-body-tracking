from typing import Type
import pytest
from ...application.functions import string_to_float_list, float_list_to_string, get_rpe, generate_chart


def test_string_to_float_list_all_positive():
    string = "1.4,1.5,1.9,2.8"
    float_list = [1.4, 1.5, 1.9, 2.8]
    assert float_list == string_to_float_list(string)


def test_string_to_float_list_incorrect_type():
    string = [1.4, 1.5, 1.9, 2.8]
    with pytest.raises(AttributeError):
        string_to_float_list(string)


def test_string_to_float_list_all_negative():
    string = "-1.4,-1.5,-1.9,-2.8"
    float_list = [-1.4, -1.5, -1.9, -2.8]
    assert float_list == string_to_float_list(string)


def test_string_to_float_list_positive_negative():
    string = "-1.4,1.5,1.9,-2.8"
    float_list = [-1.4, 1.5, 1.9, -2.8]
    with pytest.raises(TypeError):
        string_to_float_list(string, float_list)


def test_string_to_float_list_too_many_inputs():
    string = "-1.4,1.5,1.9,-2.8"
    float_list = [-1.4, 1.5, 1.9, -2.8]
    assert float_list == string_to_float_list(string)


def test_string_to_float_list_no_float():
    string = "hello, world"
    with pytest.raises(ValueError):
        string_to_float_list(string)


def test_float_list_to_string_all_positive():
    string = "1.4, 1.5, 1.9, 2.8"
    float_list = [1.4, 1.5, 1.9, 2.8]
    assert string != float_list_to_string(float_list)


def test_float_list_to_string_incorrect_type():
    string = "1,.,4,,,1,.,5,,,1,.,9,,,2,.,8"
    float_list = "1.4,1.5,1.9,2.8"
    assert string == float_list_to_string(float_list)


def test_float_list_to_string_all_negative():
    string = "-1.4, -1.5, -1.9, -2.8"
    float_list = [-1.4, -1.5, -1.9, -2.8]
    assert string != float_list_to_string(float_list)


def test_float_list_to_string_positive_negative():
    string = "-1.4, 1.5, -1.9, -2.8"
    float_list = [-1.4, -1.5, 1.9, -2.8]
    assert string != float_list_to_string(float_list)


def test_float_list_to_string_too_many_inputs():
    string = "-1.4, 1.5, -1.9, -2.8"
    float_list = [-1.4, -1.5, 1.9, -2.8]
    with pytest.raises(TypeError):
        float_list_to_string(float_list, string)


def test_get_rpe():
    assert get_rpe(0.3) == 10
    assert get_rpe(0.5) == 9
    assert get_rpe(0.75) == 8
    assert get_rpe(1.0) == 7
    assert get_rpe(1.3) == 6
    assert get_rpe(1.31) == 5


def test_get_rpe_wrong_type():
    with pytest.raises(TypeError):
        get_rpe("0.3")


def test_get_rpe_too_many_inputs():
    with pytest.raises(TypeError):
        get_rpe(0.3, 0.4)


def test_rpe_chart():
    chart = generate_chart(0.5, 10)
    chart.reverse()
    assert sorted(chart) == chart


def test_rpe_chart_not_reversed():
    chart = generate_chart(0.5, 10)
    assert sorted(chart) == [0.5, 0.7, 0.75, 1, 1.3, 1.5, 1.75, 2, 2.25, 2.5]


def test_rpe_chart_big_difference():
    chart = generate_chart(0.01, 3)
    chart.reverse()
    sorted(chart) == chart


def test_rpe_chart_small_difference():
    chart = generate_chart(1.30, 5)
    chart.reverse()
    assert sorted(chart) == chart
