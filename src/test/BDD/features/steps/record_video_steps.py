from behave import *
from selenium.common.exceptions import NoSuchElementException

# from selenium.webdriver.common.keys import Keys
# import sys

APP_TEST_ENV = "https://bbtestdeploy.herokuapp.com/"


@given("A user loads the Capture Video page")
def loadVideo(context):
    context.browser.get(APP_TEST_ENV)
    elem = context.browser.find_element_by_id("login")
    elem.click()
    context.browser.implicitly_wait(10)
    context.browser.find_element_by_id("username-input").send_keys("z")
    context.browser.find_element_by_id("password-input").send_keys("welcome12")
    context.browser.find_element_by_class_name("createAccount").click()
    context.browser.find_element_by_id("video").click()
    context.browser.find_element_by_id("liftselector").click()
    context.browser.find_element_by_xpath("//*[contains(text(), \"Deadlift\")]").click()
    context.browser.find_element_by_id("colours").click()
    context.browser.find_element_by_xpath("//*[contains(text(), \"Green\")]").click()
    context.browser.find_element_by_class_name("selectliftcolour").click()
    context.browser.find_element_by_id("recp").click()

@given("The user clicks start recording")
def record(context):
    context.browser.find_element_by_id("startbtn").click()
    context.browser.implicitly_wait(10)


@given("The user clicks stop recording")
def stopRecord(context):
    context.browser.find_element_by_id("stopbtn").click()


@when("The user clicks submit")
def submitVideo(context):
    context.browser.find_element_by_id("saveVid").click()


@then("The page loads with their captured video")
def capturedVideoLoad(context):
    try:
        context.browser.find_element_by_tag_name("video")
    except NoSuchElementException:
        return False
    return True