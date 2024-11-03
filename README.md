# DataScience-Models-and-Deployment : Chapter-3 Deploying Machine Learning Models with FLASK Web Development

## Create a model file (model.sav)
```py
import sklearn # pip install scikit-learn
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

# การอ่านไฟล์ .csv
dataset = pd.read_csv('/content/sample_data/gradingsystem_training.csv')
x = dataset.iloc[:, 2].values.reshape(-1, 1)  # ค่าเกรดเฉลี่ยสะสม
y = dataset.iloc[:, 3].values.reshape(-1, 1)  # ค่าเกรดเฉลี่ยเทอม

# การแบ่งข้อมูล
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)

# การสร้างแบบจำลองการถดถอยเชิงเส้นอย่างง่าย
regression_model = LinearRegression()
regression_model.fit(x_train, y_train)

# บันทึกแบบจำลองที่สร้างแล้ว เก็บไว้ในไฟล์ model.sav
filename = "model.sav"
joblib.dump(regression_model, filename)
newmodel = joblib.load(filename)
newmodel.predict([[30]])
```

```shell
array([[2.88205188]])
```

## Directory Structure
```shell
your_project_folder/
│
├── app.py                   # Flask application
├── model.sav                # Serialized model file
└── templates/
    └── index.html           # HTML template for the form
```

## Flask Application Code (app.py)
```py
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
```

## HTML Template (templates/index.html)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>การทำนายเกรดของเครื่องประมวลผลวิชาคณิตศาสตร์</title>
    <style>
        * {
            padding: 0;
            margin: 0;
            box-sizing: border-box;
            background-color: #8bc0f5;
        }
        .div2 {
            position: absolute;
            top: 50%;
            left: 50%;
            padding: 30px;
            transform: translate(-50%, -50%);
            border: 1px solid black;
        }
    </style>
</head>
<body>
    <div class="div2">
        <h3>การทำนายคะแนนวิชาคณิตศาสตร์ตามแบบทดสอบผลคะแนนสุดท้าย</h3> <br><br>
        <form action="/" method="POST">
            <label for="math">กรุณากรอกผลคะแนนคณิตศาสตร์:</label>
            <input type="text" id="math" name="math" required><br><br>
            <div class="center">
                <input type="submit" value="ทำนายผล" />
            </div>
        </form>
        <br />
        {% if result is not none %}
        <p>เกรดเฉลี่ยสะสมคือ : {{ result }}</p>
        {% endif %}
    </div>
</body>
</html>
```

## Run the Flask Application
```shell
python app.py
```

![01](/01.png)

##  Testing the Application
1. Open a web browser and go to `http://localhost:5000`.
2. Enter a math score in the input field and submit.
3. The predicted GPA will be displayed below the form.

![02](/02.png)

---

[Goto main](https://github.com/KOPE-Solution/DataScience-Models-and-Deployment/tree/main)
