# This is the file that implements a flask server to do inferences. It's the file that you will modify
# to implement the prediction for your own algorithm.

from __future__ import print_function

import os, sys, stat
import json
import shutil
import flask
from flask import Flask, jsonify
import glob

from fastai.imports import *
from fastai.vision import *


MODEL_PATH = '/opt/ml/'
TMP_MODEL_PATH = '/tmp/ml/model'
DATA_PATH = '/tmp/data'
MODEL_NAME = '' 

IMG_FOR_INFERENCE = os.path.join(DATA_PATH, 'image_for_inference.jpg')

# in this tmp folder, image for inference will be saved
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH, mode=0o755,exist_ok=True)

# creating a model folder in tmp directry as opt/ml/model is read-only and 
# fastai's load_learner requires to be able to write.
if not os.path.exists(TMP_MODEL_PATH):
    os.makedirs(TMP_MODEL_PATH, mode=0o755,exist_ok=True)
    #print(str(TMP_MODEL_PATH) + ' has been created')
    os.chmod(TMP_MODEL_PATH, stat.S_IRWXG)
	
if os.path.exists(MODEL_PATH):
    model_file = glob.glob('/opt/ml/model/*.pkl')[0]
    path, MODEL_NAME = os.path.split(model_file)
    #print('MODEL_NAME holds: ' + str(MODEL_NAME))
    shutil.copy(model_file, TMP_MODEL_PATH)

def write_test_image(stream):
    with open(IMG_FOR_INFERENCE, "bw") as f:
        chunk_size = 4096
        while True:
            chunk = stream.read(chunk_size)
            if len(chunk) == 0:
                return
            f.write(chunk)

            
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ClassificationService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance."""
        return load_learner(path=TMP_MODEL_PATH) #default model name of export.pkl 

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        
        learn = cls.get_model()
        return learn.predict(input) 

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ClassificationService.get_model() is not None  

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():

    write_test_image(flask.request.stream) #receive the image and write it out as a JPEG file.
    
    # Do the prediction
    img = open_image(IMG_FOR_INFERENCE)
    predictions = ClassificationService.predict(img) #predict() also loads the model
    
    #print('predictions: ' + str(predictions[0]) + ', ' + str(predictions[1]))
    
    # Convert result to JSON
    return_value = { "predictions": {} }
    return_value["predictions"]["class"] = str(predictions[0])
    print(return_value)

    return jsonify(return_value) 
