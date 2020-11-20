#!/usr/bin/python
import pandas as pd
from numpy import genfromtxt, savetxt
from sklearn.model_selection import train_test_split

# Load the dataset
raw_data = pd.read_csv('./data/loadcurves.csv', header=None)

labels = genfromtxt('./data/euclidean_clustering_result.csv', delimiter=',')

X_train, X_test, y_train, y_test = train_test_split(
    raw_data, labels,
    test_size=0.33, random_state=42
)

try:
    # save curves data in format of pandas dataframe
    X_train.to_csv('./data/train_curves.csv', index=False, header=False)
    X_test.to_csv('./data/test_curves.csv', index=False, header=False)
    
    # save labels data in format of numpy array
    savetxt(
        "./data/train_clustering_result.csv",
        y_train,
        delimiter=","
    )
    savetxt(
        "./data/test_clustering_result.csv",
        y_test,
        delimiter=","
    )

except RuntimeError:
    print("Saving results failed")
else:
    print("Successfully split and saved results")