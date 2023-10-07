import streamlit as st
from pydantic import BaseModel
import joblib

class PredictionInput(BaseModel):
    PetalLengthCm: float
    PetalWidthCm: float

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("randomForestIris.pkl")

model = load_model()

def predict(input_data):
    prediction = model.predict([[input_data.PetalLengthCm, input_data.PetalWidthCm]])
    return prediction[0]

def main():
    st.title("Position Lease Checker")
    st.write("Enter Petal Length and Petal Width to predict.")

    petal_length = st.text_input("Petal Length (cm):")
    petal_width = st.text_input("Petal Width (cm):")

    if petal_length and petal_width and st.button("Predict"):
        input_data = PredictionInput(PetalLengthCm=float(petal_length), PetalWidthCm=float(petal_width))
        prediction = predict(input_data)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
