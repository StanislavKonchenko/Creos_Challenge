#!/usr/bin/python

import pickle
import sys
from datetime import datetime
from os import path as os_path
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd
import utils
from numpy import genfromtxt, random
from tslearn.neural_network import TimeSeriesMLPClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset

if __name__ == '__main__':
    
    start_global = timer()
    
    print("#############################################################################################")
    
    print("Multi-Layer Perceptron classification method")
    
    # Set a seed to ensure determinism
    random.seed(216)
    
    #############################################################################################
    # Load data, normalise it, train (fit) the model
    #############################################################################################
    
    # get current working directory
    working_dir_path = Path.cwd()
    sys.path.append(str(working_dir_path))
    
    # Load the dataset
    raw_data = pd.read_csv(os_path.join(working_dir_path, "./data/train_curves.csv"), header=None)
    time_series_train = to_time_series_dataset(raw_data)
    
    labels_train = genfromtxt(os_path.join(working_dir_path, "./data/train_clustering_result.csv"), delimiter=',')
    
    # Normalize the time series
    time_series_train = TimeSeriesScalerMinMax().fit_transform(time_series_train)
    
    # Define the model
    mlp_nn_model = TimeSeriesMLPClassifier(
        hidden_layer_sizes=(64, 64),
        random_state=1
    )
    
    # fit the model using the training data
    mlp_nn_model.fit(time_series_train, labels_train)
    
    #############################################################################################
    # save model and visualization
    #############################################################################################
    
    print("#############################################################################################")
    
    # return string with current datetime
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    try:
        # save model to models folder
        filename = os_path.join(working_dir_path, './models/mlp_nn.pickle')
        pickle.dump(mlp_nn_model, open(filename, 'wb'))
        
    except RuntimeError:
        print("Saving results failed")
    else:
        print("Successfully saved results")
    
    #############################################################################################
    # Calculate classification rate
    #############################################################################################
    
    # Load the dataset
    X_test = pd.read_csv(os_path.join(working_dir_path, "./data/test_curves.csv"), header=None)
    X_test = to_time_series_dataset(X_test)
    
    y_test = genfromtxt(os_path.join(working_dir_path, "./data/test_clustering_result.csv"), delimiter=',')
    
    # calculate classification rate
    score = mlp_nn_model.score(X_test, y_test)
    
    print("#############################################################################################")
    
    print("Correct classification rate: % s" % score)
    
    #############################################################################################
    
    # obtain and print executing time of data processing stage to console,
    timer_tick = utils.get_time_tick(start_global)
    
    print("#############################################################################################")
    
    print("Elapsed time: % s" % timer_tick)
    
    print("#############################################################################################")
    
    #############################################################################################
