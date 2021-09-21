@picklift @feature
Feature: Picklift Successful
  As a User,
  I want to be able to pick a lift and colour successfully,
  So I can upload video.

  Scenario: User Picks a lift 
    Given A user loads the homepage
    And The user navigates to the login page
    And The user logs in
    And The user navigates to Capture Video page
    When The user chooses their lift
    Then The upload or record page shows
