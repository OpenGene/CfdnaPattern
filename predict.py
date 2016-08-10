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
from sklearn.externals import joblib

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

def main():
    if sys.version_info.major >2:
        print('python3 is not supported yet, please use python2')
        sys.exit(1)

    (options, args) = parseCommand()

    data, samples = preprocess(options)

    model = joblib.load(options.model_file) 

    labels = model.predict(data)

    print(data)

    plot_data_list(samples, data, "predict_fig")

    for i in xrange(len(samples)):
        print(get_type_name(labels[i]) + ": " + samples[i])

if __name__  == "__main__":
    main()