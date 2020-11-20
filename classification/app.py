#!/usr/bin/python
# app.py
import json
import pickle

from flask import Flask
from flask_restful import Api, Resource, reqparse
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.shapelets import LearningShapelets
from os import path as os_path
from pathlib import Path

from tslearn.utils import to_time_series_dataset, to_time_series

APP = Flask(__name__)
API = Api(APP)

# Load models from disk
k_nn_model = KNeighborsTimeSeriesClassifier.from_pickle('./models/k_nn.pickle')
shapelets_model = LearningShapelets.from_pickle('./models/learning_shapelets.pickle')

working_dir_path = Path.cwd()

filename = os_path.join(working_dir_path, './models/mlp_nn.pickle')
mlp_nn_model = pickle.load(open(filename, 'rb'))

filename = os_path.join(working_dir_path, './models/gak_svm.pickle')
gak_svm_model = pickle.load(open(filename, 'rb'))


class Classify(Resource):
    
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('classifier_type')
        parser.add_argument('input_data')

        # creates dict
        args = parser.parse_args()
        arr = json.loads(args['input_data'])
        # convert input to array
        arr1 = np.array([elem for singleList in arr for elem in singleList])
        X_check = to_time_series_dataset(arr1)

        # Use loaded model to make predictions
        if args['classifier_type'] == "mlp_nn":
            y_pred = mlp_nn_model.predict(X_check)
        elif args['classifier_type'] == "shapelets":
            y_pred = shapelets_model.predict(X_check)
        elif args['classifier_type'] == "k_nn":
            y_pred = k_nn_model.predict(X_check)
        elif args['classifier_type'] == "gak_svm":
            y_pred = gak_svm_model.predict(X_check)
        
        out = {'Class': int(y_pred[0])}
        
        return out, 200


API.add_resource(Classify, '/classify')

if __name__ == '__main__':
    APP.run(debug=True, port='5080')
