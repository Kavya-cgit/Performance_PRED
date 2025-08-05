from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("rf_employee_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features=pickle.load(open("feature_columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST"])
def submit():
   
     try:
        form_data = request.form.to_dict()
        input_data = []

        # Convert values
        for col in features:
            value = form_data.get(col, 0)
            try:
                input_data.append(float(value))
            except:
                input_data.append(0)

        # Reshape and predict
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)
        label = encoder.inverse_transform(prediction)[0]
        return render_template("submit.html", prediction=label)

     except Exception as e:
      return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
