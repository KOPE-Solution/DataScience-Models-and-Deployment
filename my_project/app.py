from flask import Flask, request, render_template
import joblib

# Load the trained model
model = joblib.load("model.sav")

# Initialize the Flask app
app = Flask(__name__)

# Define the home route to render the HTML form
@app.route('/')
def home():
    return render_template("index.html")

# Define the prediction route to process form input and return a result
@app.route('/', methods=['POST'])
def predict():
    try:
        # Get the input value from the form
        math_score = float(request.form['math'])
        
        # Predict using the loaded model
        prediction = model.predict([[math_score]])
        
        # Extract the result (assuming it's a 2D array)
        result = round(prediction[0][0], 2)
        
        # Render the result in the template
        return render_template("index.html", result=result)
    
    except Exception as e:
        # In case of an error, return a message
        return f"Error occurred: {e}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
