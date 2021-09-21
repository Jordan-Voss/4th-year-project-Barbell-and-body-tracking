from behave import *
from selenium.common.exceptions import NoSuchElementException

# from selenium.webdriver.common.keys import Keys
# import sys

APP_TEST_ENV = "https://bbtestdeploy.herokuapp.com/"


@given("the user loads the Stash page")
def loadStash2(context):
    context.browser.get(APP_TEST_ENV)
    context.browser.find_element_by_id("stash").click()


@when("the user clicks a video")
def navigateVideo(context):
    context.browser.find_element_by_link_text('z/bar/Squat-167.mp4').click()


@then("the stash info page shows")
def navigateVideo(context):
    element = context.browser.find_element_by_id('totalscore')
    assert element.text == 'You Scored 1/3!'
