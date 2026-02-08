from fastapi import FastAPI
import joblib

app = FastAPI(title="Spam Detector API")

model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "Spam Detector API Running"}

@app.get("/predict")
def predict(text: str):
    result = model.predict([text])[0]
    return {
        "input": text,
        "prediction": result
    }