@logout @feature
Feature: Logout Successful
  As a User,
  I want to be able to log out successfully,
  So I can stop using the app.

  Scenario: User Logs out In Successfully
    Given A user loads the homepage
    And The user navigates to the login page
    And The user logs in
    When The user navigates to logout page
    Then The Log in screen shows 
