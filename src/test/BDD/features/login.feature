@login @feature
Feature: Login Successful
  As a User,
  I want to be able to log in successfully,
  So I can use the app.

  Scenario: User Logs In Successfully
    Given A user loads the homepage
    And The user navigates to the login page
    When The user logs in
    Then The Logged in screen shows
