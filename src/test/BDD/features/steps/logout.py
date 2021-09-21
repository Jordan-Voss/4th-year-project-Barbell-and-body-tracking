from behave import *
from selenium.common.exceptions import NoSuchElementException

# from selenium.webdriver.common.keys import Keys
# import sys

APP_TEST_ENV = "https://bbtestdeploy.herokuapp.com/"

@given("A user loads the homepage")
def loadHome(context):
    context.browser.get(APP_TEST_ENV)


@given(u"The user logs in")
def login(context):
    context.browser.implicitly_wait(10)
    context.browser.find_element_by_id("username-input").send_keys("z")
    context.browser.find_element_by_id("password-input").send_keys("welcome12")
    context.browser.find_element_by_class_name("createAccount").click()

@when("The user navigates to logout page")
def navigateLogout(context):
    context.browser.get(APP_TEST_ENV)
    context.browser.find_element_by_id("logout").click()


@then("The Log in screen shows")
def loginShow(context):
    element = context.browser.find_element_by_tag_name('h1')
    assert element.text == 'Log In'
