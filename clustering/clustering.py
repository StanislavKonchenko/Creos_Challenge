#!/usr/bin/python

import getopt
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape
from tslearn.clustering import KernelKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from timeit import default_timer as timer
from datetime import timedelta
from string import Formatter


class Arguments:
    """ Arguments class represents and manipulates input arguments """
    
    def __init__(
            self,
            number_clusters=3,
            input_file='./data/loadcurves.csv',
            normalise_data=False,
            standardise_data=False,
            euclidean_clustering=False,
            dtw_clustering=False,
            soft_dtw_clustering=False,
            k_shape_clustering=False,
            gak_clustering=False,
    
    ):
        """ Instantiate arguments """
        self.number_clusters = number_clusters
        self.input_file = input_file
        self.normalise_data = normalise_data
        self.standardise_data = standardise_data
        self.euclidean_clustering = euclidean_clustering
        self.dtw_clustering = dtw_clustering
        self.soft_dtw_clustering = soft_dtw_clustering
        self.k_shape_clustering = k_shape_clustering
        self.gak_clustering = gak_clustering


# format function to return human friendly timedelta value
def str_from_delta(t_delta, fmt='{H:02}h {M:02}m {S:02}s', input_type='timedelta'):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the str_from_delta() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The input_type argument allows t_delta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid input_type strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """
    
    # Convert t_delta to integer seconds.
    if input_type == 'timedelta':
        remainder = int(t_delta.total_seconds())
    elif input_type in ['s', 'seconds']:
        remainder = int(t_delta)
    elif input_type in ['m', 'minutes']:
        remainder = int(t_delta) * 60
    elif input_type in ['h', 'hours']:
        remainder = int(t_delta) * 3600
    elif input_type in ['d', 'days']:
        remainder = int(t_delta) * 86400
    elif input_type in ['w', 'weeks']:
        remainder = int(t_delta) * 604800
    
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values) + ' %sms' % str(t_delta.microseconds)[:-3]


# function to load data
def import_data():
    r_d = pd.read_csv('./data/loadcurves.csv', header=None)
    return r_d


# function to calculate, format and return string with human friendly elapsed time
def get_time_tick(start_timer):
    end_timer = timer()
    return str_from_delta(timedelta(seconds=end_timer - start_timer))


# function to create matplotlib figure axe (subplot) with some parameters,
# draw time series attributed to one of the clusters, and subplot label
def create_figure_axes(
        figure, g_spec, n_row, index,
        x_lim, y_lim_min, y_lim_max,
        f_data, c_data, color_label
):
    f_ax = figure.add_subplot(g_spec[n_row, index])
    f_ax.set_xlim(0, x_lim)
    f_ax.set_ylim(y_lim_min, y_lim_max)
    cases = 0
    for d in f_data[c_data == index]:
        cases += 1
        f_ax.plot(d.ravel(), "k-", alpha=.2)
    f_ax.text(0.55, 0.95,
              'Cluster %d (%d)' % (index + 1, cases),
              transform=plt.gca().transAxes,
              color=color_label
              )
    return f_ax


def main(argv):
    # define global timer to obtain global execution time
    start_global = timer()
    
    # define globals variables
    global euclidean_clustered_data, \
        dtw_clustered_data, \
        soft_dtw_clustered_data, \
        k_shape_clustered_data, \
        gak_clustered_data
    
    #############################################################################################
    # Input arguments parsing
    #############################################################################################
    
    # define help message
    help_message = \
        'clustering.py -h \n\n' \
        'usage: clustering.py [-c <number_clusters>] [-i <input_file>] [-ansEDSKG] \n' \
        'by default: processing input data (without any sampling)' \
        '(euclidean, dtw, soft-dtw and GAK k-means, k-shape)\n' \
        'options list: \n' \
        '  -c / --clusters <number_clusters>  # set number of clusters (default 3) \n\n' \
        '  -i / --ifile <input_file>          # set input filename \n' \
        '  -n / --normalise                   # normalise input data \n' \
        '  -s / --standardise                 # standardise input data \n\n' \
        '  -a / --all                         # perform all 5 implemented methods of clustering: \n' \
        '                                       euclidean, dtw, soft-dtw, gak k-means and k-shape\n' \
        '  -E / --euclidean                   # perform euclidean k-means clustering \n' \
        '  -D / --dtw                         # perform dtw k-means clustering \n' \
        '  -S / --soft-dtw                    # perform soft-dtw k-means clustering \n' \
        '  -K / --k-shape                     # perform k-shape clustering \n' \
        '  -G / --gak                         # perform GAK k-means clustering \n'
    
    # Create new object to save arguments
    i_args = Arguments()
    
    # number of rows in plot to create correct number of subplots
    # default = 3 (raw data plus distribution histograms)
    n_rows_plot = 3
    
    # define validation rules for arguments
    try:
        opts, args = getopt.getopt(
            argv,
            "hc:i:nsaEDSKG",
            [
                "help",
                "clusters=",
                "ifile=",
                "normalise",
                "standardise",
                "all",
                "euclidean",
                "dtw",
                "soft-dtw",
                "k-shape",
                "gak"
            ]
        )
    except getopt.GetoptError:
        print(help_message)
        sys.exit(2)
    
    # parse arguments
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)
            sys.exit()
        elif opt in ("-c", "--clusters"):
            i_args.number_clusters = arg
        elif opt in ("-i", "--ifile"):
            i_args.input_file = arg
        elif opt in ("-n", "--normalise"):
            i_args.normalise_data = True
        elif opt in ("-s", "--standardise"):
            i_args.standardise_data = True
        elif opt in ("-E", "--euclidean"):
            n_rows_plot += 1
            i_args.euclidean_clustering = True
        elif opt in ("-D", "--dtw"):
            n_rows_plot += 1
            i_args.dtw_clustering = True
        elif opt in ("-S", "--soft-dtw"):
            n_rows_plot += 1
            i_args.soft_dtw_clustering = True
        elif opt in ("-K", "--k-shape"):
            n_rows_plot += 1
            i_args.k_shape_clustering = True
        elif opt in ("-G", "--gak"):
            n_rows_plot += 1
            i_args.gak_clustering = True
        elif opt in ("-a", "--all"):
            n_rows_plot = 8
            i_args.euclidean_clustering = True
            i_args.dtw_clustering = True
            i_args.soft_dtw_clustering = True
            i_args.k_shape_clustering = True
            i_args.gak_clustering = True
    
    # normalise maximum number of subplots levels
    n_rows_plot = 8 if n_rows_plot > 8 else n_rows_plot
    
    #############################################################################################
    # Raw data processing stage
    #############################################################################################
    
    # set style to matplotlib plot
    mpl.style.use('seaborn')
    
    # set seed value and seed the generator
    seed = 0
    numpy.random.seed(seed)
    
    # import data and print first 5 rows
    raw_data = import_data()
    print(raw_data.head())
    
    # convert raw data to the format which can be used by tslearn
    # (3-d dimensional array)
    # BUILT functionality: adjust all time series to one size
    # (NaN values are appended to the shorter ones)
    formatted_data = to_time_series_dataset(raw_data)
    
    # print shape of new array
    print(formatted_data.shape)
    
    # obtain number of measuring
    n_measuring = formatted_data.shape[1]
    
    # define figure, grid_spec to create layout of the plot
    fig = plt.figure(constrained_layout=True)
    grid_spec = fig.add_gridspec(
        n_rows_plot,
        i_args.number_clusters
    )
    
    # set A4 size to figure
    fig.set_size_inches(8.5, 11.75)
    
    # setup count of layers of subplots
    count_layer = 3
    # setup first subplot and draw raw time series
    f_ax_raw_data = fig.add_subplot(grid_spec[:2, :])
    
    for xx in formatted_data:
        f_ax_raw_data.plot(xx.ravel(), alpha=.2)
    
    formatted_data_min = formatted_data.min()
    formatted_data_max = formatted_data.max()
    # draw title for chart with min and max values
    f_ax_raw_data.set_title('Raw Data (min = %.2f, max = %.2f)' %(formatted_data_min, formatted_data_max))

    # obtain and print executing time of data processing stage to console,
    timer_tick = get_time_tick(start_global)
    plt.ion()
    plt.show()
    
    print("Raw data processing time: %s" % timer_tick)
    
    #############################################################################################
    # Data preprocessing stage
    #############################################################################################
    
    start = timer()
    
    # Convert NaNs to value predicted by interpolation
    # linearly interpolate for NaN/NaNs
    n_nan_changes = 0
    for ind in range(formatted_data.shape[0]):
        mask = numpy.isnan(formatted_data[ind])
        n_nan_changes += mask.sum()
        formatted_data[ind][mask] = numpy.interp(
            numpy.flatnonzero(mask),
            numpy.flatnonzero(~mask),
            formatted_data[ind][~mask]
        )
    print("%d NaN values was/were interpolated" % n_nan_changes)
    
    # Scaling
    # to know should we use normalization or standardization, we need to see
    # the distribution of values.
    
    # take random 3 measuring for each case to draw histograms
    random_indexes = numpy.random.choice(n_measuring, i_args.number_clusters, replace=False)
    
    # create new arrays with values of randomly chosen measurements
    histogram_data = formatted_data[:, random_indexes]
    
    # draw histograms
    for i_histogram in range(i_args.number_clusters):
        f_ax_histogram = fig.add_subplot(grid_spec[2, i_histogram])
        f_ax_histogram.hist(
            histogram_data[:, i_histogram],
            bins=25, density=True
        )
        
        f_ax_histogram.text(0.55, 0.98,
                            'Measurement #%d' % random_indexes[i_histogram],
                            transform=plt.gca().transAxes,
                            color="navy"
                            )
        if i_histogram == 1:
            preprocessing = ''
            if i_args.normalise_data:
                preprocessing += "normalised"
                if i_args.standardise_data:
                    preprocessing += " and standardised"
            elif i_args.standardise_data:
                preprocessing += "standardised"

            preprocessing = '' if preprocessing == '' else "(data will be %s)" % preprocessing
            f_ax_histogram.set_title(
                "Distributions histograms %s" % preprocessing,
                color='navy', y=1, pad=14
            )
    
    # if no processing data option chosen continue with raw data
    processed_data = formatted_data
    
    # since for this concrete challenge data the distributions are more/less
    # Gaussian/Normal we can use standardization
    
    # normalize data: Min-Max scaling ranging between 0 and 1
    if i_args.normalise_data:
        processed_data = TimeSeriesScalerMinMax().fit_transform(processed_data)
        print("Data was normalised")
    
    # standardize data: scaling technique where the values are centered around
    # the mean with a unit standard deviation
    if i_args.standardise_data:
        processed_data = TimeSeriesScalerMeanVariance().fit_transform(processed_data)
        print("Data was standardised")
    
    # obtain max value of data (to be used in visualization subplots)
    max_data = processed_data.max() * 1.2
    min_data = processed_data.min() * 1.2
    
    timer_tick = get_time_tick(start)
    print("#############################################################################################")
    print("Data processing stage elapsed time: %s" % timer_tick)
    
    #############################################################################################
    # Implementing Euclidean k-means clustering algorithm
    #############################################################################################
    
    if i_args.euclidean_clustering:
        
        start = timer()
        print("Euclidean k-means")
        
        # define parameters of the model of the algorithm
        k_means_euclidean = TimeSeriesKMeans(
            n_clusters=i_args.number_clusters,
            verbose=True,
            random_state=seed,
            n_jobs=4
        )
        
        # calculate cluster's label array
        euclidean_clustered_data = k_means_euclidean.fit_predict(processed_data)
        
        # draw subplots with attributed clusters of time series as well as
        # cluster centers' lines
        for i_cluster in range(i_args.number_clusters):
            f_ax_euclidean = create_figure_axes(fig, grid_spec, count_layer, i_cluster,
                                                n_measuring, min_data, max_data,
                                                processed_data, euclidean_clustered_data, 'tab:blue')
            
            f_ax_euclidean.plot(
                k_means_euclidean.cluster_centers_[i_cluster].ravel(),
                "tab:green"
            )
            
            if i_cluster == 1:
                middle_axis = f_ax_euclidean
        
        # increment count of filled layer of subplots
        count_layer += 1
        
        # obtain processing time, print it to console and
        # add it to the title of the series of subplots
        timer_tick = get_time_tick(start)
        middle_axis.set_title(
            "Euclidean $k$-means (%s)" % timer_tick,
            color='tab:green', y=1, pad=14
        )
        print("#############################################################################################")
        print("Euclidean k-means time processing: %s" % timer_tick)
        
    #############################################################################################
    # Implementing DTW k-means clustering algorithm
    # use dtw (Dynamic Time Warping Distance) metric to calculate
    # distance between means
    #############################################################################################
    
    if i_args.dtw_clustering:
        
        start = timer()
        print("DTW k-means")
        k_means_DTW = TimeSeriesKMeans(n_clusters=i_args.number_clusters,
                                       n_init=3,
                                       metric="dtw",
                                       verbose=True,
                                       max_iter_barycenter=10,
                                       random_state=seed,
                                       n_jobs=6
                                       )
        dtw_clustered_data = k_means_DTW.fit_predict(processed_data)
        
        for i_cluster in range(i_args.number_clusters):
            f_ax_dtw = create_figure_axes(fig, grid_spec, count_layer, i_cluster,
                                          n_measuring, min_data, max_data,
                                          processed_data, dtw_clustered_data, 'tab:blue')
            
            f_ax_dtw.plot(
                k_means_DTW.cluster_centers_[i_cluster].ravel(),
                "tab:red"
            )
            if i_cluster == 1:
                middle_axis = f_ax_dtw

        # increment count of filled layer of subplots
        count_layer += 1
        
        timer_tick = get_time_tick(start)
        middle_axis.set_title(
            "DTW $k$-means (%s)" % timer_tick,
            color='tab:red', y=1, pad=14
        )
        print("#############################################################################################")
        print("DTW k-means time processing: %s" % timer_tick)
    
    #############################################################################################
    # Implementing soft DTW k-means clustering algorithm
    # use soft dtw (Dynamic Time Warping Distance) metric to calculate
    # distance between means
    #############################################################################################
    
    if i_args.soft_dtw_clustering:
        
        start = timer()
        print("Soft-DTW k-means")
        k_means_soft_DTW = TimeSeriesKMeans(n_clusters=i_args.number_clusters,
                                            metric="softdtw",
                                            metric_params={"gamma": .025},
                                            verbose=True,
                                            random_state=seed,
                                            n_jobs=6
                                            )
        soft_dtw_clustered_data = k_means_soft_DTW.fit_predict(processed_data)
        
        for i_cluster in range(i_args.number_clusters):
            f_ax_soft_dtw = create_figure_axes(fig, grid_spec, count_layer, i_cluster,
                                               n_measuring, min_data, max_data,
                                               processed_data, soft_dtw_clustered_data, 'tab:blue')
            
            f_ax_soft_dtw.plot(
                k_means_soft_DTW.cluster_centers_[i_cluster].ravel(),
                "tab:purple"
            )
            
            if i_cluster == 1:
                middle_axis = f_ax_soft_dtw

        # increment count of filled layer of subplots
        count_layer += 1

        timer_tick = get_time_tick(start)
        middle_axis.set_title(
            "Soft-DTW $k$-means (%s)" % timer_tick,
            color='tab:purple', y=1, pad=14
        )
        print("#############################################################################################")
        print("Soft-DTW k-means time processing: %s" % timer_tick)
    
    #############################################################################################
    # Implementing k-Shape clustering algorithm
    #############################################################################################
    
    if i_args.k_shape_clustering:
        
        start = timer()
        print("K-Shape")
        k_shape = KShape(n_clusters=i_args.number_clusters,
                         verbose=True,
                         random_state=seed
                         )
        k_shape_clustered_data = k_shape.fit_predict(processed_data)
        
        for i_cluster in range(i_args.number_clusters):
            
            min_axe_value = min(min_data, k_shape.cluster_centers_[i_cluster].ravel().min())
            max_axe_value = max(max_data, k_shape.cluster_centers_[i_cluster].ravel().max())
            
            f_ax_k_shape = create_figure_axes(fig, grid_spec, count_layer, i_cluster,
                                              n_measuring, min_axe_value, max_axe_value,
                                              processed_data, k_shape_clustered_data, 'tab:blue')
            
            f_ax_k_shape.plot(
                k_shape.cluster_centers_[i_cluster].ravel(),
                "tab:orange"
            )
            
            if i_cluster == 1:
                middle_axis = f_ax_k_shape

        # increment count of filled layer of subplots
        count_layer += 1
        
        timer_tick = get_time_tick(start)
        middle_axis.set_title(
            "$K$-Shape (%s)" % timer_tick,
            color='tab:orange', y=1, pad=14
        )
        print("#############################################################################################")
        print("K-Shape time processing: %s" % timer_tick)
    
    #############################################################################################
    # Implementing Global Alignment kernel k-means clustering algorithm
    # since kernel is used, there is no centroid of the cluster
    #############################################################################################
    
    if i_args.gak_clustering:
        
        start = timer()
        print("GAK-k-means")
        gak_k_means = KernelKMeans(n_clusters=i_args.number_clusters,
                                   kernel="gak",
                                   kernel_params={"sigma": "auto"},
                                   n_init=10,
                                   verbose=True,
                                   random_state=seed,
                                   n_jobs=6
                                   )
        
        gak_clustered_data = gak_k_means.fit_predict(processed_data)
        
        for i_cluster in range(i_args.number_clusters):
            f_ax_gak_k_means = create_figure_axes(fig, grid_spec, count_layer, i_cluster,
                                                  n_measuring, min_data, max_data,
                                                  processed_data, gak_clustered_data, 'tab:blue')
            
            if i_cluster == 1:
                middle_axis = f_ax_gak_k_means

        # increment count of filled layer of subplots
        count_layer += 1
        
        timer_tick = get_time_tick(start)
        middle_axis.set_title(
            "Global Alignment kernel $k$-means (%s)" % timer_tick,
            color='tab:cyan', y=1, pad=14)
        print("#############################################################################################")
        print("GAK k-means time processing: %s" % timer_tick)
    
    #############################################################################################
    
    # return string with current datetime
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # define the name of the directory to be created
    path = "./out/%s" % now

    print("#############################################################################################")
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    
    try:
        # save figure as pdf to out folder
        fig.savefig("./out/%s/visual_result.pdf" % now)
    
        # save clustering results
        if i_args.euclidean_clustering:
            numpy.savetxt(
                "./out/%s/euclidean_clustering_result.csv" % now,
                euclidean_clustered_data,
                delimiter=","
            )
        if i_args.dtw_clustering:
            numpy.savetxt(
                "./out/%s/dtw_clustering_result.csv" % now,
                dtw_clustered_data,
                delimiter=","
            )
        if i_args.soft_dtw_clustering:
            numpy.savetxt(
                "./out/%s/soft_dtw_clustering_result.csv" % now,
                soft_dtw_clustered_data,
                delimiter=","
            )
        if i_args.k_shape_clustering:
            numpy.savetxt(
                "./out/%s/k_shape_clustering_result.csv" % now,
                k_shape_clustered_data,
                delimiter=","
            )
        if i_args.gak_clustering:
            numpy.savetxt(
                "./out/%s/gak_clustering_result.csv" % now,
                gak_clustered_data,
                delimiter=","
            )
    except RuntimeError:
        print("Saving results failed")
    else:
        print("Successfully saved results in the path %s " % path)

    #############################################################################################
    
    # obtain and print global executing time
    timer_tick = get_time_tick(start_global)
    print("#############################################################################################")
    print("All algorithms elapsed time: % s" % timer_tick)
    
    #############################################################################################

    # render and show plot
    # plt.show()
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to finish.")
    print("#############################################################################################")
    #############################################################################################


if __name__ == "__main__":
    main(sys.argv[1:])
