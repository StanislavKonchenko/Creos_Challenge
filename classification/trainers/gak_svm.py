#!/usr/bin/python
import pickle
import sys
from datetime import datetime
from os import path as os_path
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import utils
from numpy import genfromtxt, random
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.svm import TimeSeriesSVC
from tslearn.utils import to_time_series_dataset

if __name__ == '__main__':

    start_global = timer()
    
    print("#############################################################################################")
    
    print("Global alignment kernel (GAK) for support vector classification method")
    
    # Set a seed to ensure determinism
    random.seed(16)
    
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
    gak_svm_classification_model = TimeSeriesSVC(
        kernel="gak",
        gamma=.1,
        n_jobs=4,
        verbose=1
    )
    
    # fit the model using the training data
    gak_svm_classification_model.fit(time_series_train, labels_train)
    
    #############################################################################################
    # Calculate classification rate
    #############################################################################################
    
    # load test data
    X_test = pd.read_csv(os_path.join(working_dir_path, "./data/test_curves.csv"), header=None)
    X_test = to_time_series_dataset(X_test)
    X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
    
    y_test = genfromtxt(os_path.join(working_dir_path, "./data/test_clustering_result.csv"), delimiter=',')
    
    # calculate classification rate
    score = gak_svm_classification_model.score(X_test, y_test)
    
    print("#############################################################################################")
    
    print("Correct classification rate: % s" % score)
    
    #############################################################################################
    # Visualise support vectors
    #############################################################################################
    
    n_classes = len(set(labels_train))
    
    fig = plt.figure(tight_layout=True)
    
    # set A4 size to figure
    fig.set_size_inches(8.5, 11.75)
    
    support_vectors = gak_svm_classification_model.support_vectors_
    for i, cl in enumerate(set(labels_train)):
        plt.subplot(n_classes, 1, i + 1)
        plt.title("Support vectors for class %d" % cl)
        for ts in support_vectors[i]:
            plt.plot(ts.ravel())
    
    #############################################################################################
    # save visualization
    #############################################################################################
    
    print("#############################################################################################")
    
    # return string with current datetime
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    try:
        # save figure as pdf to out folder
        fig.savefig(os_path.join(working_dir_path, "./out/visual_result_gak_svm_%s.pdf" % now))
        
        # save the model to disk
        filename = os_path.join(working_dir_path, './models/gak_svm.pickle')
        pickle.dump(gak_svm_classification_model, open(filename, 'wb'))
        
    except RuntimeError:
        print("Saving results failed")
    else:
        print("Successfully saved results")
    
    #############################################################################################
    
    # obtain and print executing time of data processing stage to console,
    timer_tick = utils.get_time_tick(start_global)
    
    print("#############################################################################################")
    
    print("Elapsed time: % s" % timer_tick)
    
    plt.show()
    input("Close the plot and press [enter] to finish.")
    
    print("#############################################################################################")
    
    #############################################################################################
