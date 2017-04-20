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
    tokenSplit = r'[\w\']+|[""!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~""\\]'
    with open('./data/javascript.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    javascript_data = ' '.join([''.join(d) for d in data_list])
    # print(len(javascript_data))

    javascript_data = re.findall(tokenSplit,javascript_data)
    javascript_data = [(' ').join([''.join(d) for d in javascript_data[x:x+100]]) for x in range(0,len(javascript_data),100)]
    print('javascript length: ',len(javascript_data))
    #Java data
    with open('./data/java.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    java_data = ' '.join([''.join(d) for d in data_list])
    # print(len(java_data))

    java_data = re.findall(tokenSplit,java_data)

        # flattened = [d for d in data for data in java_data]
        # print(flattened[0])
    java_data = [(' ').join([''.join(d) for d in java_data[x:x+100]]) for x in range(0,len(java_data),100)]
    # print(java_data[0])
    print('java length: ',len(java_data))

    #CPP data
    with open('./data/cpp.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    cpp_data = ' '.join([''.join(d) for d in data_list])
    # print(len(cpp_data))

    cpp_data = re.findall(tokenSplit,cpp_data)
    # print(len(cpp_data))

        # flattened = [d for d in data for data in cpp_data]
        # print(flattened[0])
    cpp_data = [(' ').join([''.join(d) for d in cpp_data[x:x+100]]) for x in range(0,len(cpp_data),100)]
    print('cpp length: ',len(cpp_data))

    # C# data
    with open('./data/csharp.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    csharp_data = ' '.join([''.join(d) for d in data_list])
    # print(len(csharp_data))

    csharp_data = re.findall(tokenSplit,csharp_data)
    # print(len(csharp_data))

        # flattened = [d for d in data for data in csharp_data]
        # print(flattened[0])
    csharp_data = [(' ').join([''.join(d) for d in csharp_data[x:x+100]]) for x in range(0,len(csharp_data),100)]
    print('csharp length: ',len(csharp_data))

    # Python data
    with open('./data/python.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    python_data = ' '.join([''.join(d) for d in data_list])
    # print(len(python_data))

    python_data = re.findall(tokenSplit,python_data)
    # print(len(python_data))

        # flattened = [d for d in data for data in python_data]
        # print(flattened[0])
    python_data = [(' ').join([''.join(d) for d in python_data[x:x+100]]) for x in range(0,len(python_data),100)]
    print('python length: ',len(python_data))

    # Ruby data
    with open('./data/ruby.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    ruby_data = ' '.join([''.join(d) for d in data_list])
    # print(len(ruby_data))

    ruby_data = re.findall(tokenSplit,ruby_data)
    # print(len(ruby_data))

        # flattened = [d for d in data for data in ruby_data]
        # print(flattened[0])
    ruby_data = [(' ').join([''.join(d) for d in ruby_data[x:x+100]]) for x in range(0,len(ruby_data),100)]
    print('ruby length: ',len(ruby_data))

    # Objective C data

    with open('./data/objc.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
    objc_data = ' '.join([''.join(d) for d in data_list])
    # print(len(objc_data))

    objc_data = re.findall(tokenSplit,objc_data)
    # print(len(objc_data))

        # flattened = [d for d in data for data in objc_data]
        # print(flattened[0])
    objc_data = [(' ').join([''.join(d) for d in objc_data[x:x+100]]) for x in range(0,len(objc_data),100)]
    print('objc length: ',len(objc_data))

    # Split by words

    x_text = javascript_data + java_data + cpp_data + csharp_data + python_data + ruby_data + objc_data
    # print(x_text)
    # Generate labels
    javascript_label = [[1, 0,0,0,0,0,0] for _ in javascript_data]
    java_label = [[0,1,0,0,0,0,0] for _ in java_data]
    cpp_label = [[0,0,1,0,0,0,0] for _ in cpp_data]
    csharp_label = [[0,0,0,1,0,0,0] for _ in csharp_data]
    python_label = [[0,0,0,0,1,0,0] for _ in python_data]
    ruby_label = [[0,0,0,0,0,1,0] for _ in ruby_data]
    objc_label = [[0,0,0,0,0,0,1] for _ in objc_data]

    y = np.concatenate([javascript_label, java_label, cpp_label,csharp_label,python_label,ruby_label, objc_label], 0)
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
