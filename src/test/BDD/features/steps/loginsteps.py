from behave import *
from selenium.common.exceptions import NoSuchElementException

# from selenium.webdriver.common.keys import Keys
# import sys

APP_TEST_ENV = "https://bbtestdeploy.herokuapp.com/"



@given("The user navigates to the login page")
def navigateLogin(context):
    context.browser.get(APP_TEST_ENV)
    elem = context.browser.find_element_by_id("login")
    elem.click()


@when(u"The user logs in")
def login(context):
    context.browser.implicitly_wait(10)
    context.browser.find_element_by_id("username-input").send_keys("test")
    context.browser.find_element_by_id("password-input").send_keys("welcome12")
    context.browser.find_element_by_class_name("createAccount").click()


@then("The Logged in screen shows")
def loginShow(context):
    element = context.browser.find_element_by_tag_name('h1')
    assert element.text == 'BBTrack'
    


#   links_div = context.browser.find_element_by_id("links")
#   assert len(links_div.find_elements_by_xpath("//div")) > 0
#   search_input = context.browser.find_element_by_name("q")
#   assert search_input.get_attribute("value") == phrase
