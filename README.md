# Fraud Detection System for Financial Transactions

## Overview
This project focuses on detecting fraudulent transactions within financial datasets. The goal was to develop a robust model capable of accurately identifying potential fraud while minimizing false positives.

## Data Preparation and Model Training
The dataset was split into training, validation, and test sets to ensure rigorous evaluation. Techniques like SMOTE and RandomUnderSampler were used to handle class imbalance, improving the model's ability to generalize. Feature selection via RandomForestClassifier highlighted the most relevant variables for prediction.

## Machine Learning Models
Multiple algorithms including Logistic Regression, Support Vector Classifier (SVC), RandomForestClassifier, and KNeighborsClassifier were trained and optimized using GridSearchCV. Evaluation metrics such as precision, recall, F1 score, and F2 score were employed to assess model performance.

## Performance and Conclusion
The SVC model emerged as the top performer, achieving high recall to effectively identify fraudulent transactions. It was deployed as a web application using Streamlit, allowing users to input transaction parameters and instantly determine the likelihood of fraud.

## Deployment
The final SVC model is deployed as a web app where users can input transaction details and receive immediate feedback on the transaction's authenticity. This solution not only enhances security measures but also aids in mitigating financial risks associated with fraudulent activities.

## Future Enhancements
Future iterations may focus on incorporating real-time transaction data and improving model interpretability for better decision-making in fraud detection scenarios.
