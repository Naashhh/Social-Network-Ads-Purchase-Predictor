import gradio as gr
import pandas as pd
import pickle

# Load trained pipeline
with open("Social_Network_RF_Pipeline.pkl", "rb") as f:
    model = pickle.load(f)


# Prediction Function
def predict_purchase(gender, age, salary):

    # Dummy User ID (required because model trained with it)
    user_id = 99999999

    # Create input dataframe
    input_df = pd.DataFrame({
        "User ID": [user_id],
        "Gender": [gender],
        "Age": [age],
        "EstimatedSalary": [salary]
    })

    # Feature Engineering (MUST match training)
    input_df["Age_Salary_Ratio"] = input_df["EstimatedSalary"] / (input_df["Age"] + 1)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        return "✅ Customer Will Purchase"
    else:
        return "❌ Customer Will NOT Purchase"


# Gradio UI
inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Age", value=30),
    gr.Number(label="Estimated Salary", value=50000)
]

app = gr.Interface(
    fn=predict_purchase,
    inputs=inputs,
    outputs="text",
    title="Social Network Ads Purchase Predictor"
)

app.launch(share=True)
