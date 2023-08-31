from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


model = joblib.load('LR_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract user input from the form
            pregnancies = float(request.form.get['pregnancies'])
            glucose = float(request.form.get['glucose'])
            blood_pressure = float(request.form.get['blood_pressure'])
            skin_thickness = float(request.form.get['skin_thickness'])
            insulin = float(request.form.get['insulin'])
            bmi = float(request.form.get['bmi'])
            diabetes_pedigree_function = float(request.form.get['diabetes_pedigree_function'])
            age = float(request.form.get['age'])

            input_data = [[
                pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age
            ]]
            prediction = model.predict(input_data)

            result = "Diabetes" if prediction[0] == 1 else "No Diabetes"

            return render_template('result.html', result=result)
        except Exception as e:
            return "An error occurred: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)
