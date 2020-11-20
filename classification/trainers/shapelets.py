#!/usr/bin/python
import sys
from datetime import datetime
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from matplotlib import cm
from numpy import genfromtxt, random, unique, meshgrid, arange, argmax
from tensorflow.keras.optimizers import Adam
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets
from tslearn.utils import to_time_series_dataset
from pathlib import Path
from os import path as os_path

import utils

if __name__ == '__main__':
    
    start_global = timer()
    
    print("#############################################################################################")
    
    print("Learning Shapelets classification method")
    
    # Set a seed to ensure determinism
    random.seed(116)
    
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
    
    # Get dimensions of the dataset
    n_time_series, time_series_size = time_series_train.shape[:2]
    n_classes = len(set(labels_train))
    
    # We will extract 2 shapelets and align them with the time series
    shapelet_sizes = {10: 2}
    
    # Define the model
    shapelet_classification_model = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                                      weight_regularizer=0.0001,
                                                      optimizer=Adam(lr=0.01),
                                                      max_iter=300,
                                                      verbose=1,
                                                      scale=False,
                                                      random_state=42)
    
    # fit the model using the training data
    shapelet_classification_model.fit(time_series_train, labels_train)
    
    #############################################################################################
    # Visualise shapelets, time series and distance transformed time series
    #############################################################################################
    
    # Plot distances in a 2D space
    distances = shapelet_classification_model.transform(time_series_train).reshape((-1, 2))
    weights, biases = shapelet_classification_model.get_weights('classification')
    
    # get color map
    color_map_viridis = cm.get_cmap('viridis', 3)
    
    # create figure to visualise shapelets, time series, distances
    fig = plt.figure(constrained_layout=True)
    
    # set A4 size to figure
    fig.set_size_inches(8.5, 11.75)
    
    # create a grid for two shapelets in the top, distances in the bottom
    # and time series in the middle
    grid_spec = fig.add_gridspec(8, 6)
    fig_shapelet_1 = fig.add_subplot(grid_spec[0, :3])
    fig_shapelet_2 = fig.add_subplot(grid_spec[0, 3:])
    fig_class_0 = fig.add_subplot(grid_spec[1, :2])
    fig_class_1 = fig.add_subplot(grid_spec[1, 2:4])
    fig_class_2 = fig.add_subplot(grid_spec[1, 4:])
    fig_distance_2d_plot = fig.add_subplot(grid_spec[2:, :])
    
    # plot two shapelets on the top
    fig_shapelet_1.plot(shapelet_classification_model.shapelets_[0])
    fig_shapelet_1.set_title('Shapelet $\mathbf{s}_1$')
    
    fig_shapelet_2.plot(shapelet_classification_model.shapelets_[1])
    fig_shapelet_2.set_title('Shapelet $\mathbf{s}_2$')
    
    # plot time series of each class to corresponding subplots
    for i, subplot in enumerate([fig_class_0, fig_class_1, fig_class_2]):
        for k, ts in enumerate(time_series_train[labels_train == i]):
            subplot.plot(ts.flatten(), c=color_map_viridis(i / 3), alpha=0.25)
            subplot.set_title('Class %i' % i)
    
    # create a scatter plot of the 2D distances for the time series of each class.
    for i, y in enumerate(unique(labels_train)):
        fig_distance_2d_plot.scatter(distances[labels_train == y][:, 0],
                                     distances[labels_train == y][:, 1],
                                     c=[color_map_viridis(i / 3)] * sum(labels_train == y),
                                     edgecolors='k',
                                     label='Class %i' % y)
    
    # Create a mesh-grid of the decision boundaries
    x_min = min(distances[:, 0]) - 0.1
    x_max = max(distances[:, 0]) + 0.1
    y_min = min(distances[:, 1]) - 0.1
    y_max = max(distances[:, 1]) + 0.1
    mesh_x, mesh_y = meshgrid(arange(x_min, x_max, (x_max - x_min) / 200),
                              arange(y_min, y_max, (y_max - y_min) / 200))
    Z = []
    for x, y in numpy.c_[mesh_x.ravel(), mesh_y.ravel()]:
        Z.append(argmax([biases[i] + weights[0][i] * x + weights[1][i] * y
                         for i in range(3)]))
    Z = numpy.array(Z).reshape(mesh_x.shape)
    
    fig_distance_2d_plot.contourf(mesh_x, mesh_y, Z / 3, cmap=color_map_viridis, alpha=0.25)
    
    fig_distance_2d_plot.legend()
    fig_distance_2d_plot.set_xlabel('$d(\mathbf{x}, \mathbf{s}_1)$')
    fig_distance_2d_plot.set_ylabel('$d(\mathbf{x}, \mathbf{s}_2)$')
    fig_distance_2d_plot.set_xlim((x_min, x_max))
    fig_distance_2d_plot.set_ylim((y_min, y_max))
    fig_distance_2d_plot.set_title('Distance transformed time series')
    
    #############################################################################################
    # save model and visualization
    #############################################################################################
    
    print("#############################################################################################")
    
    # return string with current datetime
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    try:
        # save figure as pdf to out folder
        fig.savefig(os_path.join(working_dir_path, "./out/visual_result_learning_shapelets_%s.pdf" % now))
        
        # save model to models folder
        shapelet_classification_model.to_pickle(os_path.join(working_dir_path, './models/learning_shapelets.pickle'))
    
    except RuntimeError:
        print("Saving results failed")
    else:
        print("Successfully saved results")
    
    #############################################################################################
    # Calculate classification rate
    #############################################################################################
    
    # load test data
    # Load the dataset
    X_test = pd.read_csv(os_path.join(working_dir_path, "./data/test_curves.csv"), header=None)
    X_test = to_time_series_dataset(X_test)
    
    y_test = genfromtxt(os_path.join(working_dir_path, "./data/test_clustering_result.csv"), delimiter=',')
    
    # calculate classification rate
    score = shapelet_classification_model.score(X_test, y_test)
    
    print("#############################################################################################")
    
    print("Correct classification rate: % s" % score)
    
    #############################################################################################
    
    # obtain and print executing time of data processing stage to console,
    timer_tick = utils.get_time_tick(start_global)
    
    print("#############################################################################################")
    
    print("Elapsed time: % s" % timer_tick)
    
    plt.show()
    input("Close the plot and press [enter] to finish.")
    
    print("#############################################################################################")
    
    #############################################################################################
