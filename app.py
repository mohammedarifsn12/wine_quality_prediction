from email.mime import application
from flask import Flask, app, flash,request,render_template 
import numpy as np
import pandas as pd
from pipeline.predict_pipeline import Custom_data,predict_pipeline
application=Flask(__name__)
app=application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=["GET","POST"])
def predict_data():
    if request.method=="GET":
        return render_template('home.html')

    else:
        data = Custom_data(
        fixed_acidity=float(request.form.get('fixed_acidity', 0)),  
        volatile_acidity=float(request.form.get('volatile_acidity', 0)),
        citric_acid=float(request.form.get('citric_acid', 0)),
        residual_sugar=float(request.form.get('residual_sugar', 0)),
        chlorides=float(request.form.get('chlorides', 0)),
        free_sulfur_dioxide=float(request.form.get('free_sulfur_dioxide', 0)),
        total_sulfur_dioxide=float(request.form.get('total_sulfur_dioxide', 0)),
        density=float(request.form.get('density', 0)),
        pH=float(request.form.get('pH', 0)),
        sulphates=float(request.form.get('sulphates', 0)),
        alcohol=float(request.form.get('alcohol', 0))
)
        
        pred_df=data.get_data_as_frame()
        print(pred_df)

        predict_pipe=predict_pipeline()
        results=predict_pipe.predict(pred_df)
        return render_template('home.html',results=results[0])
from flask import make_response

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)