from behave import *
from selenium.common.exceptions import NoSuchElementException

# from selenium.webdriver.common.keys import Keys
# import sys

APP_TEST_ENV = "https://bbtestdeploy.herokuapp.com/"



@given("The user navigates to the register page")
def navigateRegister(context):
    context.browser.get(APP_TEST_ENV)
    elem = context.browser.find_element_by_id("register")
    elem.click()


@when(u"The user fills in their information")
def register(context):
    context.browser.implicitly_wait(10)
    context.browser.find_element_by_id("fname_input").send_keys("mister")
    context.browser.find_element_by_id("lname_input").send_keys("test")
    context.browser.find_element_by_id("uname_input").send_keys("test")
    context.browser.find_element_by_id("email_input").send_keys("alwaysometing@gmailcom")
    context.browser.find_element_by_id("pw_input").send_keys("welcome12")
    context.browser.find_element_by_id("pw2_input").send_keys("welcome12")
    context.browser.find_element_by_class_name("createAccount").click()
