# from behave import *
# from selenium.common.exceptions import NoSuchElementException

# # from selenium.webdriver.common.keys import Keys
# # import sys

# APP_TEST_ENV = "https://bbtestdeploy.herokuapp.com/"

# @given("The user loads the recordorupload page")
# def navigateRecordorupload(context):
#     context.browser.get(APP_TEST_ENV)
#     elem = context.browser.find_element_by_id("select")
#     elem.click()

# @given("The user navigates to upload a file ")
# def navigateRecordorupload(context):
#     context.browser.get(APP_TEST_ENV)
#     elem = context.browser.find_element_by_id("Upload Squat File")
#     elem.click()


# #   links_div = context.browser.find_element_by_id("links")
# #   assert len(links_div.find_elements_by_xpath("//div")) > 0
# #   search_input = context.browser.find_element_by_name("q")
# #   assert search_input.get_attribute("value") == phrase
