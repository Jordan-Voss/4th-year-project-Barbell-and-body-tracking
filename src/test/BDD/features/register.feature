@register @feature
Feature: Register Account
    As a User,
    I want to be able to register,
    So I can use the features of the application.
    
    Scenario: User Registers Account A Video
        Given The user navigates to the register page
        When The user fills in their information
        When The user logs in
        Then The Logged in screen shows