# Employee-Attrition
Project Documentation: Employee Attrition Prediction

The goal of this project was to predict employee attrition and provide actionable insights for the HR/business team to mitigate risks. The process involved careful handling of imbalanced data, feature engineering, model selection, and threshold optimization to balance business objectives with model performance.

1. Model Choice: XGBoost
XGBoost was selected due to its high predictive accuracy, robustness to multicollinearity, and ability to handle missing values efficiently. It is particularly effective for tabular datasets with a mix of numerical and categorical features and allows fine-tuning of hyperparameters to optimize performance.

2. Handling Imbalanced Data: scale_pos_weight
The target variable (Attrition) was heavily imbalanced, with far fewer employees leaving than staying. To address this, scale_pos_weight was applied in XGBoost, assigning a higher weight to the minority class (attrition = 1). This adjustment helped the model pay more attention to predicting attrition correctly without sacrificing overall accuracy.

3. Threshold Adjustment for Optimal Business Solution
Default classification thresholds (0.5) often fail to capture rare but critical events like employee attrition. By tuning the decision threshold based on precision-recall trade-offs, the final model achieved a balanced detection of high-risk employees while reducing false positives. This threshold optimization ensured the solution aligned with business priorities—identifying genuine attrition risks while minimizing unnecessary interventions.

Outcome:
The final XGBoost model, combined with scale_pos_weight and threshold tuning, provides actionable, reliable predictions. HR can focus retention efforts on employees most likely to leave, leading to cost savings, improved workforce stability, and informed strategic decision-making.

The screenshot of the app is uploaded in the repository.

Check out the app using this link : https://employee-attrition007.streamlit.app/
