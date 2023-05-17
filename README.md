# regression-project

# classification_churn_project

# Churn Drivers

Project Description:
    
    Zillow is a listing agency, trader, ibuyer, and data aggregator of real estate properties that provides estimates based on current market conditions but to be an effective trader or properties, it needs to be able to properly predict prices.
    
## Project Goal:

     Identify feature(s) that drive value
     Use those features/drivers of value to develop a model that would predict those values of single family home
     The information could be used to retain clientele and maybe even increase Telco's client base
    
## Initial Thoughts 

    My initial hypothesis is square footage, bathroom count, and location are the most important predictive factors.
   
## The Plan

     Acquire data from sql zillow database
         Pull relevant data columns only using a SQL Command
         Convert to DataFrame
         Look at the dataset
         Confirm understaning of all vairbles
         Create a data dictionary
     Prepare data
         Identify nulls/Nan and replace them with the mean or get rid of that column if there are too many nulls/Nans
         Identify columns with duplicate values and get rid of them
         Change column names to make them readable
         Split data between X and Y
         Identify outliers, and get rid of them
     Explore
         Split into Train, Validate, Test
         Start Visualizing Data
         Select most relevant and interesting visulaizations for final report
         Find out which features have the strongest relationship with the target variable
         Create hypothesis
         Create models
     Model
         Choose Evaluation Metric
         Baseline eval metric
         Fit and score model with Train
         Fit and score model with Validate
         Select best model
             Fit and score model with Test

     
## Explore data

     Answer the following initial questions:
        Does bathroom count affect price?
        Does Sq Ft affect price?
        Does bedroom affect price?
            
## Develop a Model to predict assessed value

      Use drivers identified in explore to build predictive models of different types
      Evaluate models on train and validate data
      Select the best model based on the highest accuracy
      Evaluate the best model on test data

 ## Data Dictionary
      
# Data Dictionary

| **Column**          | **Description**                                           |
|---------------------|-----------------------------------------------------------|
| **Value**           | Value of home assessed by county tax office               |
| **Bathrooms**       | Number of bathrooms on the propery.                       |
| **Bedrooms**        | Number of bedrooms on the property                        |
| **Sq Ft**           | Square footage of the lot of land of the property         |
| **County**          | County the property is in.                                |
| **Date**            | Date of sale of the property                              |
| **Car Garage**      | Number of cars that can fit in the garage.                |

## Steps to Reproduce

    1. Clone this repo
    2. Acquire the data from CodeUp, Telco_churn
    3. Put the data in the file containing the cloned repo
    4. Run notebook
    
## Takeaways and Conclusions

Bathroom count is the greatest indicator for home value among my features. .

## Recommendations

Offer discounts to senior citizens to reduce churn, and family bundle discounts to fortify the lower churn rate of those with partners and dependents.