import gradio as gr
import numpy as np
import joblib
import pandas as pd

# -----------------------------
# Load model parameters
# -----------------------------
scaler = joblib.load("scaler.pkl")         # Your fitted StandardScaler
weights = np.load("weights.npy")           # Your trained weights (theta)
bias = np.load("bias.npy").item()          # Use .item() if it's a single-element np array

# -----------------------------
# Sigmoid function
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_diabetes_risk(age, hba1c, glucose, gender, hypertension, heart_disease, smoking):
    # Encode gender and smoking
    gender_val = 1 if gender == "Male" else 0
    smoking_val = 1 if smoking == "Smoking" else 0

    # Interaction terms
    age_hba1c = age * hba1c
    age_glucose = age * glucose

    # Construct feature vector
    normal_features = np.array([[
    age, hba1c, glucose,
    age_hba1c, age_glucose
    ]])

    # Scale the features using the training scaler
    normal_features_df = pd.DataFrame(normal_features, columns=["age", "HbA1c_level", "blood_glucose_level", "age_hba1c", "age_glucose"])
    features_scaled = scaler.transform(normal_features_df).flatten()
    final_features = np.array([
                               features_scaled[0],
                               hypertension,
                               heart_disease,
                               features_scaled[1],
                               features_scaled[2],
                               gender_val,
                               smoking_val,
                               features_scaled[3],
                               features_scaled[4]]).reshape(1,-1)

    # Risk score
    z = np.dot(final_features, weights) + bias
    print(z)
    risk = sigmoid(z)[0]
    print(risk)
    risk_pct = round(risk * 100, 2)

    # Return formatted output
    if risk >= 0.8:
        msg = f"⚠️ High Risk: {risk_pct}% chance of diabetes"
    elif risk >= 0.5:
        msg = f"⚠️ Moderate Risk: {risk_pct}% chance of diabetes"
    else:
        msg = f"✅ Low Risk: {risk_pct}% chance of diabetes"

    return msg
    
interface = gr.Interface(
    fn=predict_diabetes_risk,
    inputs=[
        gr.Slider(0, 80, value=40, label="Age"),
        gr.Slider(3.5, 9.0, value=5.5, label="HbA1c Level"),
        gr.Slider(80, 300, value=120, label="Blood Glucose Level"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio([0, 1], label="Hypertension (0=No, 1=Yes)"),
        gr.Radio([0, 1], label="Heart Disease (0=No, 1=Yes)"),
        gr.Radio(["No Smoking", "Smoking"], label="Smoking Status")
    ],
    outputs="text",
    title="🩺 Diabetes Risk Score Calculator",
    description="Enter patient details to predict the risk of diabetes using a logistic regression model.",
)

interface.launch(share=True)