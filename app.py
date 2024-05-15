import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import DataPreprocessor, Predictor

from flask import Flask, request, render_template


app=Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict_in_inference():
    # if the method is get the returning to index.html
    if request.method=="GET":
        return render_template("predict_page.html")

    else:
        # retrieving entered values
        gender=request.form['gender']
        race_ethnicity=request.form['race_ethnicity']
        parental_level_of_education=request.form['parental_level_of_education']
        lunch=request.form['lunch']
        test_preparation_course=request.form['test_preparation_course']
        reading_score=request.form['reading_score']
        writing_score=request.form['writing_score']

        # transforming data
        data_preprocessor=DataPreprocessor(gender,race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)

        transformed_input=data_preprocessor.preprocess_input()

        # getting prediction
        model=Predictor()
        result=model.predict(transformed_input)
        
        
        return render_template("predict_page.html", result=result[0])


# running application
if(__name__)=="__main__":
    app.run(host="127.0.0.1", port="5000", debug=True)