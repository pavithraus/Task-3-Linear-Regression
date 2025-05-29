## Linear_Regression_modelling Uing Housing Price dataset

This project applies **Simple and Multiple Linear Regression** techniques to predict housing prices based on various features from the Housing price dataset.
---

## Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---
## Dataset Used

- **Dataset:** -https://github.com/pavithraus/Task-3-Linear-Regression/blob/main/Housing.csv
- **Attributes include:**
  - area, bedrooms, bathrooms, stories, parking, furnishing status, air conditioning, etc.
  - target variable: **price**

---
## Objective

- Load and explore the housing price dataset.
- Clean and preprocess the data (including handling categorical variables).
- Apply Simple Linear Regression (with a single feature).
- Apply Multiple Linear Regression (with all relevant features).
- Evaluate model performance using error metrics (MAE, MSE, R²).
- Visualize regression results and coefficients.
- Predict housing prices for custom input data.

---
## Key Tasks

###  1. Data Preprocessing
- Load the dataset and handle missing values (if any)
- Encode categorical variables using **one-hot encoding**

###  2. Model Building
- **Simple Linear Regression**: Using `area` to predict `price`
- **Multiple Linear Regression**: Using all features to predict `price`

###  3. Model Evaluation
- Metrics:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - R² Score

###  4. Visualization
- Scatter plot with regression line (Simple Linear Regression)
- Bar chart of feature coefficients (Multiple Linear Regression)

### 5. Predict price for New features
- create New housing features
- Predict the price of the new data

---

##  The Program Works

This program walks through the entire process of predicting housing prices using both **Simple** and **Multiple Linear Regression**. It includes **data loading**, **preprocessing**, **model training**, **evaluation**, and **visualization**.

###  Load the Dataset
- Loads `Housing.csv`, which includes features like area, bedrooms, bathrooms, and more.
- Target variable is `price`.

###  Preprocess the Data
- Categorical columns are converted to numeric using one-hot encoding.
- Ensures all input features are suitable for regression models.

###  Define Features and Target
- `Simple Linear Regression`: uses only `'area'` as input.
- `Multiple Linear Regression`: uses all encoded and numerical features.

###  Split the Data
- Training: 80%, Testing: 20% using `train_test_split`.

###  Train the Linear Regression Model
- Trains models using `LinearRegression()` for both simple and multiple feature sets.

###  Make Predictions
- Predictions are made on the test data and compared with actual prices.

###  Evaluate the Model
- Models are evaluated using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
    
-Simple Linear Regression Evaluation Metrics:
  -Mean Absolute Error (MAE): 1474748.13
  -Mean Squared Error (MSE): 3675286604768.19
  -R² Score: 0.2729
  
- Multiple Linear Regression Evaluation Metrics:
  -Mean Absolute Error (MAE): 970043.40
  -Mean Squared Error (MSE): 1754318687330.66
  -R² Score: 0.6529

###  Visualize the Results
- Simple regression: Scatter plot with regression line.
- ![simple linear](https://github.com/user-attachments/assets/b7a9f81f-0a88-4de4-b529-6810b58ff007)
  
- Multiple regression: Bar plot of feature coefficients.
- ![Multi Linear](https://github.com/user-attachments/assets/58dbd8ac-53b9-4f44-8456-3cb66557caa4)

### Interpret the Coefficients
- Intercept and feature coefficients help explain how each feature influences the price.

### Predict Custom House Price
- Create a new sample input using property features.
- Preprocess it to match model input structure.
- Predict and display the house price.

---

## Results & Interpretation

- Simple Linear Regression shows the effect of just one feature (`area`) on the price.
- Multiple Linear Regression provides better predictions using all features.
- Evaluation metrics and plots help assess performance and interpret model behavior.
- Predict and display the house price for new input property features

---
## About

- The Housing price prection project (task) is completed by Intern- Pavithra
- Tool used to build the project: colab
- learning tool: Google webesite(geek for geeks,w3school, and Ai)
- Thanks to **Elevate Labs**
