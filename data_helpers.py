import numpy as np
import re
import itertools
from collections import Counter
import csv


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with open('./data/javascript.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    javascript_data = [d[0] for d in data_list]

    with open('./data/java.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    java_data = [d[0] for d in data_list]

    with open('./data/java.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    cpp_data = [d[0] for d in data_list]


    # Split by words
    x_text = javascript_data + java_data + cpp_data
    # Generate labels
    javascript_label = [[1, 0,0] for _ in javascript_data]
    java_label = [[0,1,0] for _ in java_data]
    cpp_label = [[0,0,1] for _ in cpp_data]
    y = np.concatenate([javascript_label, java_label, cpp_label], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
