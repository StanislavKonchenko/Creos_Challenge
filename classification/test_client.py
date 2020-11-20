#!/usr/bin/python
import json

import requests

import sys
from datetime import datetime
from os import path as os_path
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd

from numpy import genfromtxt, random, take
from tslearn.utils import to_time_series_dataset

if __name__ == '__main__':
    # Load the dataset
    X_test = pd.read_csv("./data/test_curves.csv", header=None)
    X_test = to_time_series_dataset(X_test)
    
    y_test = genfromtxt("./data/test_clustering_result.csv", delimiter=',')
    
    # choose random value from test samples
    random_index = random.randint(0, y_test.shape[0] - 1)
    test_data = X_test[random_index]
    
    list_values = test_data.tolist()
    json_str = json.dumps(list_values)
    
    # localhost and the defined port + endpoint
    url = 'http://127.0.0.1:5080/classify'
    
    body = {
        "classifier_type": "mlp_nn",
        "input_data": json_str
    }
    
    response = requests.post(url, data=body)
    
    # compare result with tested one
    print("Clustered tested class: % s" % int(take(y_test, random_index)))
    print("Classified by API class: % s" % response.json())
