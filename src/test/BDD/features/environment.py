from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from chromedriver_py import binary_path

BROWSERSTACK_URL = "https://jordanvoss2:MbKnB9ukUSeAmQqYpJmD@hub-cloud.browserstack.com/wd/hub"


def before_scenario(context, scenario):
    if "feature" in context.tags:
        context.browser = webdriver.Remote(
        command_executor=BROWSERSTACK_URL,
        desired_capabilities={
        "os_version": "Catalina","resolution": "1920x1080",
        "browser": "Chrome",
        "browser_version": "latest-beta",
        "os": "OS X",
        "name": f"Behaviour Driven Development {context.tags}: MacOS Chrome" })
    context.browser.implicitly_wait(10)


def after_scenario(context, scenario):
  if "feature" in context.tags:
    context.browser.quit()
