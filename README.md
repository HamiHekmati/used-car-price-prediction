### Used Car Price Prediction

This project aims to build robust machine learning models to predict the selling price of used cars based on real-world data. The workflow covers all critical stages of a data science pipeline, including data cleaning, feature engineering, exploratory data analysis (EDA), model development, and performance evaluation.

### Overview

Predicting the price of used cars is a classic regression problem with many real-world challenges, including missing values, categorical features, skewed distributions, and the presence of outliers. In this project, multiple regression algorithms were compared, and advanced techniques such as log transformation and feature engineering were applied to improve predictive accuracy.

### Getting Started
To run this project on your own machine, begin by cloning or downloading the repository. Make sure you have Python 3.12.3 (or a compatible Python 3.x version) installed. Install all necessary dependencies using the included requirements.txt file:

```bash
pip install -r requirements.txt
 ```
Once the environment is set up, open the Jupyter notebook file [used_cars_price_prediction.ipynb](https://github.com/HamiHekmati/used-car-price-prediction/blob/main/used_cars_price_prediction.ipynb) in Jupyter Notebook, JupyterLab, or upload it to Google Colab for a cloud-based workflow. Run all notebook cells in sequence to walk through the full data science process, from data cleaning and feature engineering to modeling and results interpretation. This setup enables you to fully reproduce the analysis, explore the code, and experiment with your own modifications.

### Dataset

Source: Kaggle:[Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data)

Description: Contains various features describing car details (brand, model, year, mileage, engine type, color, accident history, and more) along with the target variable (price).

### Project Workflow

The project began with data cleaning, which included handling missing values using techniques such as fillna() for both categorical and numerical columns, removing or treating outliers, particularly in the price column and performing necessary data type conversions to ensure monetary and numeric columns were in the correct format.

Next, exploratory data analysis (EDA) was conducted to gain insight into the dataset. This step involved creating univariate and bivariate visualizations to better understand the distribution of car prices and their relationships with various features, as well as performing correlation analysis and examining feature importance.

Feature engineering followed, where new variables such as car age, mileage per year, a luxury brand indicator, and a recent model flag were created to enrich the dataset. One-hot encoding was applied to transform categorical variables into a format suitable for machine learning models.

In the model development phase, several algorithms were implemented and compared, including Linear Regression, Random Forest, Gradient Boosting, K-Nearest Neighbors, and XGBoost. A log transformation was also applied to the price variable to address skewness and stabilize variance.

Finally, model evaluation was performed using metrics such as MAE, RMSE, and R² score. Models were compared on both the original and log-transformed targets, and model performance was visualized to aid in interpretation and selection of the best approach.

### Key Features & Techniques

Comprehensive Data Preprocessing:
Managed missing data and outliers to prepare a clean dataset for modeling.

Custom Feature Engineering:
Introduced domain-inspired features (car age, luxury indicator, mileage per year) to improve model accuracy.

Model Comparison:
Benchmarked classical and ensemble methods, and tuned hyperparameters for improved performance.

Log-Transformation:
Applied log scaling to the target variable to mitigate skewness and improve prediction reliability.

Clear Visual Communication:
Included informative visualizations and markdown commentary to guide readers through the workflow.

### Results

Best Model:
XGBoost on log-transformed price achieved the highest R² score (~0.82), outperforming other approaches.

Random Forest (log scale):
RMSE ≈ 0.39, R² ≈ 0.76

XGBoost (log scale):
RMSE ≈ 0.34, R² ≈ 0.82

Full metric comparisons and result plots are available in the notebook.


### Contact

For questions, feedback, or collaboration, reach out via [LinkedIn](https://www.linkedin.com/in/hami-hekmati-399932154/) or open an issue in this repository.

This project was developed by Hami Hekmati as a data science portfolio project.
