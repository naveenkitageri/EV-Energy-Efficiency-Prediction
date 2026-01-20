# EV-Energy-Efficiency-Prediction
Built an end-to-end ML pipeline to predict EV energy efficiency using XGBoost, including EDA, Feature engineering and model evaluation 

- This project predicts **Energy Efficiency (km/kWh)** of Electric Vechicle using various Machine Learning algorithms and selecting the best performance model.

# Project Objective 
- To build regression model that accurately predicts EV energy efficiency based on :
- motor power
- vehicle class
- make and model

# Dataset
-source: EV Energy Efficiency Dataset (kaggel Dataset)
Taget variable:
**Energy Efficiency (km/kWh)**

feature variable:
- Motor (kW)
- Recharge time (h)
- Make
- Model
- Vehicle class

## Workflow 
1. Data loading
2. Exploratory data analysis
3. Outlier handling
4. Feature engineering (one-hot-encoding)
5. train test split (80/20)
6. Feature scaling
7. Model traning
8. Model evaluation
9. Cross validation
10. Best model selection
11. Model saving

# Model performance (r2_score)
model                  |     R2 score
1.Linear Regression    |   0.81
2.Decision Tree        |   0.91
3. Random Forest       |   0.92
4. Gradient Boost      |   0.85
5. XGBoost             |   0.92
6. SVM                 |   0.81
7. KNN                 |   0.29

# Final Model 
- Selected model : **XGBoost Regression**
- Reason : Highest R2 score
- Saved Using : joblib 
