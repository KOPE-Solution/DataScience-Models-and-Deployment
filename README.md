# DataScience-Models-and-Deployment : Chapter-2 Decision Tree Classification for Prediction of Graduation

## การทำนายผลนักศึกษาใหม่ระดับปริญญาโท จะสำเร็จการศึกษาภายใน 2 ปี

### 1) โจทย์ปัญหาและที่มา
- ในแต่ละปีมีนักศึกษาระดับปริญญาโทที่ตกค้าง ไม่สำเร็จการศึกษาภายในกำหนดเวลา 2 ปี เมื่อนักศึกษาตกค้างหรือไม่สำเร็จการศึกษาภายในกำหนดเวลา จะก่อให้เกิดปัญหาตามมา อาทิเช่น ค่าใช้จ่ายของนักศึกษา ภาระงานที่ปรึกษา เป้าหมายของสาขา และมหาวิทยาลัย
- ดังนั้นเมื่อมีนักศึกษาใหม่ หรือผู้สนใจเข้าไปทำการสมัครเรียนป.โท เข้ามาในระบบ ที่อาศัยหลักการเรียนรู้ของเครื่อง สำหรับวิเคราะห์เชิงทำนายว่านักศึกษาใหม่ จะสำเร็จการศึกษาภายใน 2 ปี หรือไม่ เมื่อระบบดังกล่าวออกมา ทางหลักสูตรสามารถแจ้งข้ออาจารย์ที่ปรึกษา แล้วไปดูแลนักศึกษาเพื่อส่งเสริมให้นักศึกษาสำเร็จการศึกษาภายในกำหนดเวลา 2 ปี

## 2) รายละเอียดข้อมูล
- BSc = จบการศึกษาปริญญาตรีวิทยาศาสตร์บัณฑิต หรือที่เกี่ยวข้อง (จบตรงกับหลักสูตร) โดย 1 = จบตรง และ 0 คือจบไม่ตรง
- Gender = เพศ: 0 คือผู้ชาย และ 1 คือผู้หญิง
- Experience = ประสบการณ์การทำงาน (ปี) โดย 1= 0-2 ปี, 2= 2-5 ปี และ 3 = มากกว่า 5 ปี
- Province = จังหวัดในประเทศไทย
- Thesis/IS = ทำวิทยานิพนธ์ (0) หรือค้นคว้าอิสระ (1)
- English Proficiency Certificate = ผลสอบภาษาอังกฤษ โดยมี = 1 และไม่มี = 0
- Result = จบการศึกษาในกำหนดเวลา โดย 0 คือจบใน 2 ปี และ 1 คือไม่จบใน 2 ปี



## 1) Import Libraries

```py
# การนำเข้า Library ที่สำคัญ
import pandas as pd  # data processing
import numpy as np  # linear algebra
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # statistical data visualization
import sklearn as sk  # machine learning model
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
```

## 2) Reas Dataset
```py
# การนำไฟล์เข้า
studentdata = pd.read_csv('/content/sample_data/Generated_Dataset.csv')
studentdata.head(10)
```

![01](/01.png)

## 3) Exploratory Data Analysis : EDA
```py
# การเตรียมข้อมูล ที่ข้อมูลจะต้องเป็นตัวเลข
all_features = [name for name in studentdata.columns if studentdata[name].dtype == 'object']
all_features
```

```shell
['Province']
```

### 3.1) Convert an Object to Int
```py
# การแปลงข้อมูลจังหวัด (Province) ให้เป็นตัวเลข
all_features = [name for name in studentdata.columns if studentdata[name].dtype == 'object']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in list(all_features):
    studentdata[i] = le.fit_transform(studentdata[i])

for x in all_features:
    print(x, "=", studentdata[x].unique())
```

```shell
Province = [2 5 1 7 6 4 0 3 8]
```

<br>
```py
# การแสดงข้อมูลจังหวัดที่ถูกแปลงเป็นตัวเลข
studentdata.head(10)
```

![02](/02.png)

### 3.2) Feature Selection with Chi-Square
```py
# การเลือกคุณลักษณะ (Feature Selection) ที่สัมพันธ์กับผลลัพธ์ (Result)
from sklearn.feature_selection import chi2
studentdata.fillna(0, inplace=True)
X = studentdata.drop('Result', axis=1)
y = studentdata['Result']
chi_scores = chi2(X, y)

# การคัดเรียงลำดับความสำคัญของคุณลักษณะจากน้อยไปหามาก
p_values = pd.Series(chi_scores[1], index=X.columns)
p_values.sort_values(ascending=True, inplace=True)
p_values
```

![03](/03.png)

<br>

```py
# การสร้างกราฟแสดงคุณลักษณะจากน้อยไปหามาก
p_values.plot.bar(figsize=(10,5), cmap="coolwarm")
plt.title('Chi-square test for feature selection', size=18)
```

```shell
Text(0.5, 1.0, 'Chi-square test for feature selection')
```

![04](/04.png)

### 3.3) Remove Some Attributes After Ffeature Selection

```py
# การลบคุณลักษณะที่ไม่สำคัญออกไปจากชุดข้อมูล
newfeature = studentdata.columns.tolist()
newfeature.remove('Province')
newfeature.remove('Gender')
newfeature.remove('EnglishProficiencyCertificate')
newfeature
```

```shell
['BSc', 'CGPA', 'Experience', 'Thesis/IS', 'Result']
```

## 4) Split Training and Testing Data Set (80:20)
```python
# การแบ่งชุดข้อมูลสำหรับการเรียนออกเป็น 80:20
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

## 5) Creating a Decision Tree Model
```py
# การสร้างแบบจำลองต้นไม้ตัดสินใจ (Decision Tree)
modelDT = DecisionTreeClassifier(random_state=43)

# การฝึกสอนข้อมูล
modelDT.fit(X_train, y_train)

# การทำนายข้อมูล
predictions = modelDT.predict(X_test)
```

## 6) Evaluation Model (Confusion Matrix)
```py
# การสร้างคอนฟังชันเมทริกซ์ เพื่อแสดงความถูกต้อง แม่นยำ และความครอบคลุมของแบบจำลอง
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
```

```shell
[[10 12]
 [14 14]]
              precision    recall  f1-score   support

           0       0.42      0.45      0.43        22
           1       0.54      0.50      0.52        28

    accuracy                           0.48        50
   macro avg       0.48      0.48      0.48        50
weighted avg       0.48      0.48      0.48        50

0.48
```

## 7) Visualizing Decision Tree

```py
# การสร้างภาพต้นไม้ตัดสินใจ
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz

cols = list(X_train.columns.values)
dot_data = tree.export_graphviz(modelDT, out_file=None, feature_names=cols)

graph = graphviz.Source(dot_data, format="png")
graph

plt.savefig('DTree.png')

graph.render("DTree_graphviz")
```

```shell
DTree_graphviz.png
<Figure size 640x480 with 0 Axes>
```

## 8) Testing the Predictive Model with Unseen Data

```py
# การสร้างการทดสอบข้อมูลใหม่
studentdata1 = pd.read_csv('/content/sample_data/student_data_sample.csv')
studentdata1.head(5)
```

![05](/05.png)

## 9) Saving Prediction Result to Exel (Display_ResultDT.csv)
```py
# การทำนายชุดข้อมูลใหม่
test_predict = modelDT.predict(X=studentdata1)

# Create a submission for Kaggle
submission = pd.DataFrame({"Result": test_predict})

# Save submission to CSV
submission.to_csv("DTreeResult.csv", index=False)
```

---