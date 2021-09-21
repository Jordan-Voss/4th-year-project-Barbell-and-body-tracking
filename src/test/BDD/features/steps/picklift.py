from behave import *
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import Select

# from selenium.webdriver.common.keys import Keys
# import sys

APP_TEST_ENV = "https://bbtestdeploy.herokuapp.com/"

@given("The user navigates to Capture Video page")
def navigateCaptureVideo(context):
    context.browser.get(APP_TEST_ENV)
    elem = context.browser.find_element_by_id("video")
    elem.click()

@when("The user chooses their lift")
def pickLift(context):
    context.browser.implicitly_wait(10)
    lift = Select(context.browser.find_element_by_id('liftselector'))
    lift.select_by_visible_text('Squat')
    clr = Select(context.browser.find_element_by_id('colours'))
    clr.select_by_visible_text('Green')
    context.browser.find_element_by_id("submitcolourlift").click()

@then("The upload or record page shows")
def uplrec(context):
    element = context.browser.find_element_by_id('uplp')
    assert element.text == 'Upload Squat File'
    element = context.browser.find_element_by_id('recp')
    assert element.text == 'Record Squat Set'
