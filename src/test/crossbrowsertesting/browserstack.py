import sys
from selenium import webdriver
from selenium.webdriver.support.ui import Select

BROWSERSTACK_URL = "https://" + str(sys.argv[1]) + ":" + str(sys.argv[2]) + "@hub-cloud.browserstack.com/wd/hub"
ENV = str(sys.argv[3])

pages = ["register", "home", "video", "login"]
caps_list = [{
  "os": "Windows",
  "os_version": "10",
  "browser": "Chrome",
  "browser_version": "80",
  "name": "Cross Browser Test: Windows 10 Chrome"
},
{
  "os_version": "Catalina",
  "resolution": "1920x1080",
  "browser": "Chrome",
  "browser_version": "latest-beta",
  "os": "OS X",
  "name": "Cross Browser Test: MacOS Chrome"
}
# {
#   "os_version": "Catalina",
#   "resolution": "1920x1080",
#   "browser": "Safari",
#   "browser_version": "13.1",
#   "os": "OS X",
#   "name": "Cross Browser Test: MacOS Safari"
# }
]

for desired_cap in caps_list:
    driver = webdriver.Remote(
      command_executor=BROWSERSTACK_URL,
      desired_capabilities=desired_cap
      )
    driver.get(ENV)
    if "Homepage" not in driver.title:
        print("Unable to load Home page")
        raise Exception("Unable to load Home page!")
    button = driver.find_element_by_id("login")
    button.click()
    if "Login" not in driver.title:
        print("Unable to load Login page")
        raise Exception("Unable to load Login page!")
    driver.find_element_by_id("username-input").send_keys("z")
    driver.find_element_by_id("password-input").send_keys("welcome12")
    driver.find_element_by_class_name("createAccount").click()
    if "Homepage" not in driver.title:
        print("Unable to load Logged In page")
        raise Exception("Unable to load Logged In page!")
    print(driver.title, ":", desired_cap["name"])
    driver.find_element_by_id("stash").click()
    if "Stash" not in driver.title:
        print("Unable to load Stash page")
        raise Exception("Unable to load Stash page!")
    print(driver.title, ":", desired_cap["name"])
    driver.find_element_by_id("video").click()
    if "Picker" not in driver.title:
        print("Unable to load Picker page")
        raise Exception("Unable to load Picker page!")
    lift = Select(driver.find_element_by_id('liftselector'))
    lift.select_by_visible_text('Squat')
    clr = Select(driver.find_element_by_id('colours'))
    clr.select_by_visible_text('Green')
    driver.find_element_by_id("submitcolourlift").click()
    if "Video" not in driver.title:
        print("Unable to load Upload/Record page")
        raise Exception("Unable to load Upload/Record page!")
    driver.quit()
