@stashinfo @feature
Feature: See stash video info
    As a User,
    I want to be able to see a past lift's information,
    So I can see how much I have progressed.
    
    Scenario: User Looks at a video in their stash   
        Given The user navigates to the login page
        And The user logs in
        And the user loads the Stash page
        When the user clicks a video
        Then the stash info page shows