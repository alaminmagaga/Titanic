from flask import Flask, render_template, request, redirect, url_for
import pickle
import  numpy as np

file=open("titanic_model.pkl","rb")
model=pickle.load(file)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/submit', methods=['POST'])
def submit():
    Pclass = float(request.form['Pclass'])
    Sex = float(request.form['Sex'])
    Age = float(request.form['Age'])
    SibSp = float(request.form['SibSp'])
    Parch = float(request.form['Parch'])
    Fare = float(request.form['Fare'])
    Embarked = float(request.form['Embarked'])
    Age_Group = float(request.form['Age Group'])

    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Age_Group]])
    prediction = model.predict(features)

    


    return render_template("result.html", Pclass=Pclass,
                            Sex=Sex, 
                            Age=Age, 
                            SibSp=SibSp, 
                            Parch=Parch, 
                            Fare=Fare,
                            Embarked=Embarked,
                            Age_Group=Age_Group,
                            prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
