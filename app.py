from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))


# Define a function for prediction
def prediction(Crop, Crop_Year, State, Area, Annual_Rainfall, Yield):
    try:
        # Create a DataFrame from the input
        input_data = pd.DataFrame({
            'Crop': [Crop],
            'Crop_Year': [Crop_Year],
            'State': [State],
            'Area': [Area],
            'Annual_Rainfall': [Annual_Rainfall],
            'Yield': [Yield]
        })

        # Preprocess the input data
        input_preprocessed = preprocessor.transform(input_data)

        # Predict using the loaded model
        predicted_output = dtr.predict(input_preprocessed)

        return predicted_output[0]

    except ValueError as e:
        return f"Error: {str(e)}"

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        Crop = request.form['Crop']
        Crop_Year = int(request.form['Crop_Year'])
        State = request.form['State']
        Area = float(request.form['Area'])
        Annual_Rainfall = float(request.form['Annual_Rainfall'])
        Yield = float(request.form['Yield'])

        # Perform prediction
        result = prediction(Crop, Crop_Year, State, Area, Annual_Rainfall, Yield)

        # Return the prediction to the template
        return render_template('index.html', prediction_text=f'Predicted Production: {result}')

    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
