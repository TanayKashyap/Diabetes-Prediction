# Diabetes Prediction Project

## Description

This project aims to predict the likelihood of diabetes based on various health indicators and demographic information. The analysis includes data loading, exploration, visualization, feature engineering, and the implementation of a weighted logistic regression model to address class imbalance.

## Visuals

The notebook includes several visualizations to help understand the data and the relationships between variables:

- Distribution of diabetic vs non-diabetic patients.
- Histograms and KDE plots of numerical features (`age`, `bmi`, `HbA1c_level`, `blood_glucose_level`).
- Count plots of categorical features (`gender`, `hypertension`, `heart_disease`, `smoking_history`) against diabetes status.
- Scatter plots exploring potential interactions between `age`, `HbA1c_level`, and `blood_glucose_level`.

## Installation

To run this notebook, you will need to have the following libraries installed in your Python environment:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels
- joblib

You can install them using pip: pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels joblib

## Usage

1.  **Upload Data:** The notebook starts by prompting you to upload the `diabetes_prediction_dataset.csv` file.
2.  **Run Cells:** Execute the cells sequentially to perform the data loading, cleaning, exploration, visualization, feature engineering, model training, and evaluation steps.
3.  **Model Evaluation:** The notebook provides metrics such as accuracy, precision, recall, and F1 score to evaluate the performance of the weighted logistic regression model. It also includes a precision-recall curve.
4.  **Explore Thresholds:** The notebook demonstrates how changing the prediction threshold can impact the precision and recall of the model.
5.  **Saved Artifacts:** The trained model's weights and bias are saved as `weights.npy` and `bias.npy`, and the `StandardScaler` is saved as `scaler.pkl` for use in a risk dashboard.
6.  You can run the dashboard with streamlit run risk_dashboard.py

## Support

If you encounter any issues or have questions, please feel free to contact the project owner.

## Roadmap
- Explore other classification algorithms (e.g., Random Forest, Gradient Boosting, SVM).
- Implement more advanced techniques for handling class imbalance (e.g., SMOTE).
- Fine-tune model hyperparameters.
- Fix the dashboard lol.
