@stash @feature
Feature: Check Stash Successful
  As a User,
  I want to be able to check stash successfully,
  So I can use review my videos.

  Scenario: User Checks Stash Successfully
    Given The user navigates to the login page
    And The user logs in
    When the user loads the Stash page
    Then a list of videos will show
