from flask import Flask , request
import numpy as np
import pickle 
from flasgger import Swagger
import pandas as pd

with open('./rf.pkl' , 'rb') as model_pkl:
    model = pickle.load(model_pkl)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    """
    s_length = request.args.get('s_length')
    s_width = request.args.get('s_width')
    p_length = request.args.get('p_length')
    p_width = request.args.get('p_width')

    prediction = model.predict(np.array([[s_length , s_width , p_length , p_width]]))
    return str(prediction)

@app.route('/predict_file',methods = ['POST'])
def predict_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get('input_file') , header=None)
    predictions = model.predict(input_data)
    return str(list(predictions)) 

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=8501)


