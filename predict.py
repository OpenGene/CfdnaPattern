import sys, os
from optparse import OptionParser
import time
from util import *
from draw import *
from feature import *
import numpy as np
from sklearn import svm, neighbors
import random
import json
import pickle

def parseCommand():
    usage = "extract the features, and train the model, from the training set of fastq files. \n\npython training.py <fastq_files> [-f feature_file] [-m model_file] "
    version = "0.0.1"
    parser = OptionParser(usage = usage, version = version)
    parser.add_option("-m", "--model", dest = "model_file", default = "cfdna.model",
        help = "specify which file stored the built model.")
    return parser.parse_args()

def preprocess(options):

    data = []
    samples = []
    fq_files = get_arg_files()

    number = 0
    for fq in fq_files:
        number += 1
        #print(str(number) + ": " + fq)

        extractor = FeatureExtractor(fq)
        extractor.extract()
        feature = extractor.feature()

        if feature == None:
            #print("======== Warning: bad feature from:")
            #print(fq)
            continue

        data.append(feature)
        samples.append(fq)

    return data, samples

def get_type_name(label):
    if label == 1:
        return "cfdna"
    else:
        return "not-cfdna"

def load_model(options):
    filename = options.model_file
    if not os.path.exists(filename):
        filename = os.path.join(os.path.dirname(sys.argv[0]), options.model_file)
    if not os.path.exists(filename):
        print("Error: the model file not found: " + options.model_file)
        sys.exit(1)
    f = open(filename, "rb")
    model = pickle.load(f)
    f.close()
    return model

def main():
    if sys.version_info.major >2:
        print('python3 is not supported yet, please use python2')
        sys.exit(1)

    (options, args) = parseCommand()

    data, samples = preprocess(options)

    model = load_model(options)

    labels = model.predict(data)

    for i in xrange(len(samples)):
        print(get_type_name(labels[i]) + ": " + samples[i])

    plot_data_list(samples, data, "predict_fig")

if __name__  == "__main__":
    main()