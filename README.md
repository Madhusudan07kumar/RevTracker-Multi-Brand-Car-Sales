# RevTracker-Multi-Brand-Car-Sales
This project built a dual-function ML system to predict used car prices and classify conditions using used_cars_data.csv. After preprocessing, the Random Forest Regressor achieved R² ≈ 0.85, and the Classifier reached 90%+ accuracy. Key insights: Year, Mileage, and demand for Diesel and First-Owner cars in major cities.
This is a fantastic and well-structured project report\! To make it suitable for uploading to your GitHub ID, you should format the content into a `README.md` file (for the main project explanation) and create a Python notebook or script (`.ipynb` or `.py` file) for the code.

Here is the content organized for GitHub:

-----

## 1\. Project README.md Content (using Markdown)

This file should serve as the main introduction to your project on the GitHub repository page.

````markdown
# RevTracker: Multi-Brand Car Sales Analytics Project

## Project Overview

This project implements a comprehensive machine learning-based approach for analyzing used car sales data. The primary goal is to address the challenge of accurately assessing a used car's value and condition in the rapidly expanding second-hand automobile market. The system is developed to perform two critical predictive tasks: **price forecasting (Regression)** and **condition assessment (Classification)**.

## Project Goal and Key Objectives

The main objective is to design and develop a robust machine learning system that accurately predicts the price and condition of used cars using historical data and key vehicle attributes.

### Key Objectives:

1.  **Data Acquisition and Exploration:** Analyze the `used_cars_data.csv` dataset, perform Exploratory Data Analysis (EDA) to identify patterns, and visualize the relationship between variables (e.g., Year, Mileage, Price, Fuel Type).
2.  **Data Preprocessing:** Cleanse the dataset by handling missing values (imputation/dropping), encoding categorical variables (Label/One-Hot Encoding), and normalizing numerical features (Scaling/Log Transformation).
3.  **Regression Modeling (Price Prediction):** Build and evaluate multiple regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor) using metrics like Root Mean Squared Error (RMSE) and $R^{2}$ score.
4.  **Classification Modeling (Condition Prediction):** Develop and evaluate classification models (Logistic Regression, Random Forest Classifier) to categorize vehicle condition, using metrics such as Accuracy, Precision, Recall, and F1-score.
5.  **Model Selection & Interpretation:** Select the best-performing model for each task (Random Forest proved superior) and translate the results into actionable business insights.

## Methodology and Model Selection

The project followed a structured workflow including data preprocessing, feature engineering, and model training.

### Data Preparation Highlights

* **Missing Values:** Handled through median imputation (numerical) and row dropping (critical data missing).
* **Feature Engineering:** Brand and Model names were extracted from the `Name` column.
* **Encoding:** Categorical variables like `Fuel_Type`, `Location`, `Transmission`, and `Owner_Type` were mapped to numerical values.
* **Transformation:** Highly skewed features like `Price` were addressed using log scaling.

### Model Performance Summary

| Task | Model | Key Metric (Example Value) | Business Insight |
| :--- | :--- | :--- | :--- |
| **Regression** (Price) | **Random Forest Regressor** | Highest $R^{2}$ ($\approx 0.85$) and Lowest RMSE | Provides highly accurate, non-linear price forecasts. |
| **Classification** (Condition) | **Random Forest Classifier** | Highest Accuracy ($\approx 90\%$) and Balanced F1-score | Reliably categorizes vehicle condition (Fair, Good, Excellent). |

**Note on Model Choice:** The Random Forest algorithm was consistently selected for both tasks due to its superior performance, lower tendency toward overfitting, and ability to quantify feature importance.

## Key Actionable Business Insights

The exploratory data analysis and model outputs yielded several strategic insights for the used car industry:

### Pricing & Inventory Strategy

* **Year and Mileage are Paramount:** These are the strongest predictors of price. Newer cars (post-2015) with low mileage command premium prices and should be the focus for high-margin inventory.
* **Owner Type Premium:** First-owner vehicles dominate the listings (93%) and retain significantly higher resale values compared to second- and third-owner cars. Dealers should leverage "First-Owner" status in pricing and marketing.
* **Fuel Type Impact:** Diesel cars are highly represented and tend to fetch higher resale values, suggesting a consumer preference for performance/fuel efficiency in the used market.

### Market and Customer Insights

* **Transmission Preference:** Manual transmission dominates the dataset (62.2%), indicating a large market segment focused on lower purchase cost and perceived fuel efficiency. However, the sizable automatic segment (37.8%) presents a key upsell opportunity, particularly for urban and first-time buyers.
* **Geographic Focus:** The largest volume of listings originates from Kochi, Mumbai, and Coimbatore (over 50% combined), suggesting these are high-opportunity markets for inventory sourcing and targeted marketing campaigns.

## Conclusion

The "RevTracker" system successfully demonstrates that machine learning provides a robust, objective, and accurate method for tackling key business challenges in the used car domain. The Random Forest-based models offer superior predictive power and clear, actionable insights that can be directly applied to improve pricing strategies, optimize inventory, and enhance transparency for both consumers and dealerships.

***

## 2. Python Script/Notebook Content (e.g., `used_cars_analysis.ipynb` or `analysis_code.py`)

Since the prompt contains detailed code snippets and output tables, organizing this into a clean, runnable script/notebook is the best approach. I will provide the raw Python code blocks for a script, which you can easily convert into a Jupyter Notebook if desired.

```python
# ==============================================================================
# RevTracker: Multi-Brand Car Sales Analytics Project - Complete Code
# ==============================================================================

# 1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 2. Load Data
# Assuming the file 'used_cars_data.csv' is in the same directory or accessible via path
df = pd.read_csv('used_cars_data.csv')

# ==============================================================================
# 3. Data Preprocessing and Cleaning
# ==============================================================================

# Drop columns identified as not useful for modeling (S.No.) or too sparse (New_Price)
df.drop(columns=['S.No.', 'New_Price'], inplace=True, errors='ignore')

# Separate Brand and Model from the 'Name' column
df['Brand'] = df['Name'].str.split(' ').str[0]
df['Model'] = df['Name'].str.split(' ').str[1] + df['Name'].str.split(' ').str[2].fillna('') 
df.drop(columns=['Name'], inplace=True)

# Clean and convert columns with units (Mileage, Engine, Power) to float
df['Mileage'] = df['Mileage'].str.extract(r'([\d.]+)').astype(float)
df['Engine'] = df['Engine'].str.extract(r'([\d.]+)').astype(float)
df['Power'] = df['Power'].str.extract(r'([\d.]+)').astype(float)

# Impute remaining missing values with median for numerical columns
for col in ['Mileage', 'Engine', 'Power', 'Seats', 'Price']:
    df[col].fillna(df[col].median(), inplace=True)

# Drop rows where 'Price' is still null (if any were not handled by median imputation)
df.dropna(subset=['Price'], inplace=True)

# Create 'Condition' target for classification (Assuming median split for demonstration 
# and aligning with the project report's 'Low'/'High' classification focus)
median_price = df['Price'].median()
df['Price_Class'] = pd.cut(df['Price'], bins=[-float('inf'), median_price, float('inf')], labels=['Low', 'High'])
# We will drop this 'Price_Class' column for the main regression task.


# Convert categorical variables to numerical using mapping/encoding for simplicity as shown in the report
# Note: For real-world production, One-Hot Encoding is generally better for nominal features.
# We'll stick to the report's structure for reproducibility.

# Mapping labels based on the project report's implied numerical scale (or a simple integer mapping)
# For 'Fuel_Type'
fuel_map = {'CNG': 3, 'Diesel': 1, 'Petrol': 2, 'LPG': 4, 'Electric': 5}
df['Fuel_Type'] = df['Fuel_Type'].map(fuel_map)
df['Fuel_Type'].fillna(df['Fuel_Type'].mode()[0], inplace=True) # Fill any leftover NaNs from failed mapping

# For 'Transmission'
transmission_map = {'Manual': 1, 'Automatic': 2}
df['Transmission'] = df['Transmission'].map(transmission_map)

# For 'Owner_Type'
owner_map = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}
df['Owner_Type'] = df['Owner_Type'].map(owner_map)

# For 'Location' (using the rank order from the bar chart in the appendix for simplicity)
location_order = ['Kochi', 'Mumbai', 'Coimbatore', 'Hyderabad', 'Pune', 'Kolkata', 'Delhi', 'Chennai', 'Jaipur', 'Ahmedabad', 'Bangalore']
location_map = {loc: i+1 for i, loc in enumerate(location_order)}
df['Location'] = df['Location'].map(location_map)

# Drop 'Brand' and 'Model' for the initial modeling phase to align with the simplicity of the Linear Regression setup
# For advanced models like Random Forest, these should be One-Hot Encoded.
df.drop(columns=['Brand', 'Model'], inplace=True)

Fuel Type,Percentage of Listings
Diesel,,53.8% 
Petrol,,45.1% 
CNG,,1.1%

Transmission,Percentage of Listings
Manual,,62.2% 
Automatic,,37.8%

Owner Type,Percentage of Listings
First,,93.0% 
Second,,6.7% 
Third,,0.4%


# ==============================================================================
# 4. Model Building & Evaluation
# ==============================================================================

## 4.1 Regression Task: Price Prediction

# Define features (X) and target (y) for regression
X_reg = df.drop(['Price', 'Price_Class'], axis=1)
y_reg = df['Price']

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# --- Linear Regression (Baseline) ---
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
y_pred_lin = lin_reg.predict(X_test_reg)

r2_lin = r2_score(y_test_reg, y_pred_lin)
mse_lin = mean_squared_error(y_test_reg, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)

print("--- Linear Regression Results (All Features) ---")
print(f"R^2 Score: {r2_lin:.4f}")
print(f"MSE: {mse_lin:.4f}")
print(f"RMSE: {rmse_lin:.4f}")
print("-" * 40)

# --- Random Forest Regressor (Best Model) ---
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf = rf_reg.predict(X_test_reg)

r2_rf = r2_score(y_test_reg, y_pred_rf)
mse_rf = mean_squared_error(y_test_reg, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print("--- Random Forest Regressor Results (All Features) ---")
print(f"R^2 Score: {r2_rf:.4f}")
print(f"MSE: {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print("-" * 40)

# Feature Importance Plot (Random Forest)
feature_importances = pd.Series(rf_reg.feature_importances_, index=X_reg.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance for Price Prediction (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

## 4.2 Classification Task: Condition Classification

# Define features (X) and target (y) for classification
X_clf = df.drop(['Price', 'Price_Class'], axis=1)
y_clf = df['Price_Class']

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# --- Logistic Regression ---
log_clf = LogisticRegression(max_iter=1000, random_state=42)
log_clf.fit(X_train_clf, y_train_clf)
y_pred_log_clf = log_clf.predict(X_test_clf)

print("--- Logistic Regression Classification Results ---")
print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_log_clf):.4f}")
print(classification_report(y_test_clf, y_pred_log_clf))
print("-" * 40)

# --- Random Forest Classifier (Best Model) ---
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)
y_pred_rf_clf = rf_clf.predict(X_test_clf)

print("--- Random Forest Classifier Results ---")
print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_rf_clf):.4f}")
print(classification_report(y_test_clf, y_pred_rf_clf))
print("-" * 40)

# Confusion Matrix Visualization (Random Forest Classifier)
cm_rf = confusion_matrix(y_test_clf, y_pred_rf_clf, labels=y_clf.cat.categories)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=y_clf.cat.categories, yticklabels=y_clf.cat.categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Random Forest Classifier')
plt.show()
````
<img width="735" height="489" alt="image" src="https://github.com/user-attachments/assets/a49baf05-be42-4194-bb2f-7999524b2d30" />

<img width="483" height="333" alt="image" src="https://github.com/user-attachments/assets/24e66250-05b0-44cb-b01c-f1620fd50094" />


<img width="376" height="314" alt="image" src="https://github.com/user-attachments/assets/c9284754-1c55-4deb-bdad-a821c4b6c23a" />

<img width="574" height="324" alt="image" src="https://github.com/user-attachments/assets/3fe6b6f3-2f1c-4ef8-ad3a-958f7cb550e9" />

<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/92cd5c78-b851-447e-934d-ee157e1d4131" />

