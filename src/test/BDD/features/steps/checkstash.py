from behave import *
from selenium.common.exceptions import NoSuchElementException

# from selenium.webdriver.common.keys import Keys
# import sys

APP_TEST_ENV = "https://bbtestdeploy.herokuapp.com/"


@when("the user loads the Stash page")
def loadStash(context):
    context.browser.get(APP_TEST_ENV)
    # elem = context.browser.find_element_by_id("stash")
    # elem.click()
    # context.browser.implicitly_wait(10)
    # context.browser.find_element_by_id("username-input").send_keys("test")
    # context.browser.find_element_by_id("password-input").send_keys("welcome12")
    context.browser.find_element_by_id("stash").click()


@then("a list of videos will show")
def navigateVideo(context):
    # context.browser.get(APP_TEST_ENV)
    try:
        context.browser.find_element_by_id("processed")
    except NoSuchElementException:
        return False
    return True
