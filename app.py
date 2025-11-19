import pandas as pd
import joblib
from flask import Flask, render_template, request
import os
import sys

# Import preprocess function
try:
    from model1 import preprocess_text
except:
    print("Cannot import preprocess_text from model1.py")
    sys.exit(1)

app = Flask(__name__)

MODEL = None
VECTORIZER = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

def load_model():
    global MODEL, VECTORIZER

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model files not found!")
        return False

    MODEL = joblib.load(MODEL_PATH)
    VECTORIZER = joblib.load(VECTORIZER_PATH)
    print("Model + Vectorizer Loaded Successfully")
    return True


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        text = request.form.get("article_text")

        if not text:
            prediction = {"status": "error", "message": "Please enter text."}
        else:
            processed = preprocess_text(text)
            vector = VECTORIZER.transform([processed])
            pred = MODEL.predict(vector)[0]

            prediction = {
                "status": "success",
                "text": text,
                "result": "TRUE" if pred == 1 else "FAKE",
                "label": int(pred)
            }

    return render_template("project.html", prediction_result=prediction)


if __name__ == "__main__":
    if load_model():
        app.run(host="0.0.0.0", port=5000)
    else:
        print("Cannot start app - model missing.")
